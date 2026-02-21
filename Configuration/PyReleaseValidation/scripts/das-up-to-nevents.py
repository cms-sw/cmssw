#!/usr/bin/env python3
import pycurl
from io import BytesIO
import ast
import subprocess
import pandas as pd
import argparse
from bs4 import BeautifulSoup
import numpy as np
import os
import json
import sys
import itertools
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] : %(message)s',
    handlers=[
        logging.FileHandler('das-up-to-nevents.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants to skip short runs and the beginning of each run
MIN_RUN_LENGTH = 40
SKIP_LUMIS = 20

## Helpers
base_cert_url = "https://cms-service-dqmdc.web.cern.ch/CAF/certification/"
base_cert_eos = "/eos/user/c/cmsdqm/www/CAF/certification/"
base_cert_cvmfs = "/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/"

def get_url_clean(url):
    logger.debug(f"Fetching URL: {url}")
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    return BeautifulSoup(buffer.getvalue(), "lxml").text

def get_lumi_ranges(i):
    logger.debug("Calculating luminosity ranges")
    result = []
    for _, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        result.append([b[0][1], b[-1][1]])
    return result

def das_key(dataset):
    logger.debug(f"Creating DAS key for dataset: {dataset}")
    return 'dataset='+dataset if "#" not in dataset else 'block='+dataset

def das_query(query):
    cmd = f"dasgoclient"
    # For cms-bot deterministic caching see cms-sw#50101
    if "JENKINS_PREFIX" in os.environ:
        cmd = f"{cmd} --limit=1000 -unique"
    cmd = f"{cmd} --query='{query}'"
    logger.debug(f"Executing DAS query: {cmd}")
    out = subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode('utf8')
    result = out.split("\n")
    logger.debug(f"Query result: {result}")
    return result

def das_file_site(dataset, site):
    out = das_query(f"file {das_key(dataset)} site={site}")
    df = pd.DataFrame(out, columns=["file"])
    return df

def das_file_data(dataset, opt=""):
    out = das_query(f"file {das_key(dataset)} {opt} | grep file.name, file.nevents")
    out = [r.split(" ") for r in out if len(r) > 0]
    out = [[r[0], r[3]] for r in out if len(r) > 3]
    df = pd.DataFrame(out, columns=["file", "events"])
    df.events = df.events.values.astype(int)
    return df

def das_lumi_data(dataset, opt=""):
    out = das_query(f"file,lumi,run {das_key(dataset)} {opt}")
    out = [r.split(" ") for r in out if len(r) > 0]
    df = pd.DataFrame(out, columns=["file", "run", "lumis"])
    return df

def das_run_events_data(dataset, run, opt=""):
    out = das_query(f"file {das_key(dataset)} run={run} {opt} | sum(file.nevents)")
    out = out[0].split(" ")[-1] if out else "0"
    return int(out)

def das_run_data(dataset, opt=""):
    out = das_query(f"run {das_key(dataset)} {opt}")
    return out

def no_intersection():
    logger.error("No intersection between JSON and dataset. Exiting.")
    print("No intersection between:")
    print(" - json   : ", best_json)
    print(" - dataset: ", dataset)
    print("Exiting.")
    sys.exit(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', default=None, help="Dataset Name (e.g. '/DisplacedJet/Run2024C-v1/RAW', may also be a block (e.g /ZeroBias/Run2024J-v1/RAW#d8058bab-4e55-45b0-abb6-405aa3abc2af)",type=str,required=True)
    parser.add_argument('--threshold','-t', help ="Event threshold per file",type=int,default=-1)
    parser.add_argument('--events','-e', help ="Tot number of events targeted",type=int,default=-1)
    parser.add_argument('--outfile','-o', help='Dump results to file', type=str, default=None)
    parser.add_argument('--pandas', '-pd',action='store_true',help="Store the whole dataset (no event or threshold cut) in a csv")
    parser.add_argument('--proxy','-p', help='Allow to parse a x509 proxy if needed', type=str, default=None)
    parser.add_argument('--site','-s', help='Only data at specific site', type=str, default=None)
    parser.add_argument('--lumis','-l', help='Output file for lumi ranges for the selected files (if black no lumiranges calculated)', type=str, default=None)
    parser.add_argument('--nogolden','-ng', action='store_true', help='Do not crosscheck the dataset run and lumis with a Golden json for data certification')
    parser.add_argument('--run','-r', help ="Target a specific run",type=int,default=None,nargs="+")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug or "JENKINS_PREFIX" in os.environ else logging.INFO)

    if args.proxy is not None:
        os.environ["X509_USER_PROXY"] = args.proxy
        logger.debug(f"Set X509_USER_PROXY to {args.proxy}")
    elif "X509_USER_PROXY" not in os.environ:
        logger.error("No X509 proxy set. Exiting.")
        print("No X509 proxy set. Exiting.")
        sys.exit(1)

    ## Check if we are in the cms-bot "environment"
    dataset   = args.dataset
    events    = args.events
    threshold = args.threshold
    outfile   = args.outfile
    site      = args.site
    lumis     = args.lumis
    runs      = args.run
    das_opt = ""

    logger.info(f"Dataset: {dataset}, Events: {events}, Threshold: {threshold}, Outfile: {outfile}, Site: {site}, Lumis: {lumis}, Runs: {runs}")

    if runs is not None:
        das_opt = "run in %s"%(str([int(r) for r in runs]))
        logger.debug(f"DAS options for runs: {das_opt}")

    if not args.nogolden:
        logger.debug("Checking for golden JSON files")

        ## get the greatest golden json
        year = dataset.split("Run")[1][2:4] # from 20XX to XX
        PD = dataset.split("/")[1]
        cert_type = "Collisions" + str(year)
        if "Cosmics" in dataset:
            cert_type = "Cosmics" + str(year)
        elif "Commisioning" in dataset:
            cert_type = "Commisioning2020"
        elif "HI" in PD:
            cert_type = "Collisions" + str(year) + "HI"

        logger.info(f"Certification type: {cert_type}")

        cvmfs_path = base_cert_cvmfs + cert_type + "/"
        eos_path = ""
        web_path = ""
        json_list_full = []
        ## if we have access to cvmfs we get from there ...
        if os.path.isdir(cvmfs_path):
            cvmfs_path = cvmfs_path + "/latest/"
            json_list_full = os.listdir(cvmfs_path)
            logger.info(f"Found JSON files in CVMFS: {json_list_full}")
        ## ... if not we try eos ...
        if len(json_list_full)==0:
            eos_path = base_cert_eos + cert_type + "/"
            if os.path.isdir(eos_path):
                json_list_full = os.listdir(eos_path)
                logger.info(f"Found JSON files in EOS: {json_list_full}")
        ## ... if not we go to the website
        if len(json_list_full)==0:
            web_path = base_cert_url + cert_type + "/"
            json_list_full = get_url_clean(web_path).split("\n")
            logger.info(f"Found JSON files on web: {json_list_full}")

        pattern = re.compile("(cert_collisions\d{4}_\d*_\d*_golden.json)(\s|$)", re.IGNORECASE)
        json_list = [match.group(1) for entry in json_list_full for match in [re.search(pattern, entry)] if match and match.group(1)]
        if len(json_list)==0:
            logger.error(f"No matching JSON files found from {source} ({path}). The full list was:\n{list_full}")
            raise RuntimeError("No matching JSON files found from {source} ({path}). The full list was:\n{list_full}".format(
                source="web" if web_path else "eos" if eos_path else "cvmfs",
                path=web_path if web_path else eos_path if eos_path else cvmfs_path,
                list_full='\n'.join(json_list_full),
            ))

        # the larger the better, assuming file naming schema
        # Cert_X_RunStart_RunFinish_Type.json
        # TODO if args.run keep golden only with right range
        run_ranges = [int(c.split("_")[-2]) - int(c.split("_")[-3]) for c in json_list]
        latest_json = np.array(json_list[np.argmax(run_ranges)]).reshape(1,-1)[0].astype(str)
        best_json = str(latest_json[0])
        logger.info(f"Selected JSON file: {best_json}")

        if not web_path:
            with open((eos_path if eos_path else cvmfs_path) + "/" + best_json) as js:
                golden = json.load(js)
        else:
            golden = get_url_clean(web_path + best_json)
            golden = ast.literal_eval(golden) #converts string to dict

        # skim for runs in input
        if runs is not None:
            for k in golden:
                if k not in args.run:
                    golden.pop(k)

        # golden json with all the lumisections
        golden_flat = {}
        for k in golden:
            R = []
            for r in golden[k]:
                # skipping short runs
                if r[1]-r[0] + 1 < MIN_RUN_LENGTH:
                    continue

                R = R + [f for f in range(r[0]+SKIP_LUMIS,r[1]+1)]
            golden_flat[k] = R

        # let's just check there's an intersection between the
        # dataset and the json
        data_runs = das_run_data(dataset)
        golden_data_runs = [r for r in data_runs if r in golden_flat]

        if (len(golden_data_runs)==0):
            no_intersection()

        # building the dataframe, cleaning for bad lumis
        golden_data_runs_tocheck = golden_data_runs

    df = das_lumi_data(dataset,opt=das_opt).merge(das_file_data(dataset,opt=das_opt),on="file",how="inner") # merge file informations with run and lumis

    df["lumis"] = [[int(ff) for ff in f.replace("[","").replace("]","").split(",")] for f in df.lumis.values]

    if not args.nogolden:
        logger.debug("Filtering data based on golden JSON")

        df_rs = []
        for r in golden_data_runs_tocheck:
            cut = (df["run"] == r)
            if not any(cut):
                continue

            df_r = df[cut]

            # jumping low event content runs
            if df_r["events"].sum() < threshold:
                continue

            # taking only fully certified files, i.e. files for which all lumis are in the golden json
            good_lumis = np.array([len([ll for ll in l if ll in golden_flat[r]]) for l in df_r.lumis])
            n_lumis = np.array([len(l) for l in df_r.lumis])
            df_rs.append(df_r[good_lumis==n_lumis])

        if (len(df_rs)==0):
            no_intersection()

        df = pd.concat(df_rs)

    df.loc[:,"min_lumi"] = [min(f) for f in df.lumis]
    df.loc[:,"max_lumi"] = [max(f) for f in df.lumis]
    df.loc[:,"n_lumis"] = df.loc[:,"max_lumi"] - df.loc[:,"min_lumi"] + 1

    df = df.sort_values(["run","n_lumis","min_lumi","max_lumi"])

    if site is not None:
        df = df.merge(das_file_site(dataset,site),on="file",how="inner")

    if args.pandas:
        df.to_csv(dataset.replace("/","")+".csv")

    if events > 0:
        logger.debug(f"Filtering files with more than {events} events")
        df = df[df["events"] <= events] #jump too big files
        df.loc[:,"sum_evs"] = df.loc[:,"events"].cumsum()
        df = df[df["sum_evs"] < events]

    files = df.file

    if lumis is not None:
        logger.debug(f"Saving luminosity ranges to {lumis}")
        lumi_ranges = { int(r) : list(get_lumi_ranges(np.sort(np.concatenate(df.loc[df["run"]==r,"lumis"].values).ravel()).tolist())) for r in np.unique(df.run.values).tolist()}

        with open(lumis, 'w') as fp:
            json.dump(lumi_ranges, fp)

    if outfile is not None:
        logger.debug(f"Saving file list to {outfile}")
        with open(outfile, 'w') as f:
            for line in files:
                f.write(f"{line}\n")
    else:
        logger.debug("Outputting file list to console")
        print("\n".join(files))

    sys.exit(0)
