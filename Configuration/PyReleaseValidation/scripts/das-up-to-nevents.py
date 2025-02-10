#!/usr/bin/env python3
import pycurl
from io import BytesIO
import pycurl
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
import json

## Helpers
base_cert_url = "https://cms-service-dqmdc.web.cern.ch/CAF/certification/"
base_cert_path = "/eos/user/c/cmsdqm/www/CAF/certification/"

def get_url_clean(url):
    
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    
    return BeautifulSoup(buffer.getvalue(), "lxml").text

def get_lumi_ranges(i):
    result = []
    for _, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        result.append([b[0][1],b[-1][1]]) 
    return result

def das_do_command(cmd):
    out = subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode('utf8')
    return out.split("\n")

def das_file_site(dataset, site):
    cmd = "dasgoclient --query='file dataset=%s site=%s'"%(dataset,site)
    out = das_do_command(cmd)
    df = pd.DataFrame(out,columns=["file"])

    return df

def das_file_data(dataset,opt=""):
    cmd = "dasgoclient --query='file dataset=%s %s| grep file.name, file.nevents'"%(dataset,opt)
    out = das_do_command(cmd)
    out = [np.array(r.split(" "))[[0,3]] for r in out if len(r) > 0]

    df = pd.DataFrame(out,columns=["file","events"])
    df.events = df.events.values.astype(int)
    
    return df

def das_lumi_data(dataset,opt=""):
    cmd = "dasgoclient --query='file,lumi,run dataset=%s %s'"%(dataset,opt)
    
    out = das_do_command(cmd)
    out = [r.split(" ") for r in out if len(r)>0]
    
    df = pd.DataFrame(out,columns=["file","run","lumis"])
    
    return df

def das_run_events_data(dataset,run,opt=""):
    cmd = "dasgoclient --query='file dataset=%s run=%s %s | sum(file.nevents) '"%(dataset,run,opt)
    out = das_do_command(cmd)[0]

    out = [o for o in out.split(" ") if "sum" not in o]
    out = int([r.split(" ") for r in out if len(r)>0][0][0])

    return out

def das_run_data(dataset,opt=""):
    cmd = "dasgoclient --query='run dataset=%s %s '"%(dataset,opt)
    out = das_do_command(cmd)

    return out

def no_intersection():
    print("No intersection between:")
    print(" - json   : ", best_json)
    print(" - dataset: ", dataset)
    print("Exiting.")
    sys.exit(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', default=None, help="Dataset Name (e.g. '/DisplacedJet/Run2024C-v1/RAW' )",type=str,required=True)
    parser.add_argument('--threshold','-t', help ="Event threshold per file",type=int,default=-1)
    parser.add_argument('--events','-e', help ="Tot number of events targeted",type=int,default=-1)
    parser.add_argument('--outfile','-o', help='Dump results to file', type=str, default=None)
    parser.add_argument('--pandas', '-pd',action='store_true',help="Store the whole dataset (no event or threshold cut) in a csv") 
    parser.add_argument('--proxy','-p', help='Allow to parse a x509 proxy if needed', type=str, default=None)
    parser.add_argument('--site','-s', help='Only data at specific site', type=str, default=None)
    parser.add_argument('--lumis','-l', help='Output file for lumi ranges for the selected files (if black no lumiranges calculated)', type=str, default=None)
    parser.add_argument('--precheck','-pc', action='store_true', help='Check run per run before building the dataframes, to avoid huge caching.')
    args = parser.parse_args()

    if args.proxy is not None:
        os.environ["X509_USER_PROXY"] = args.proxy
    elif "X509_USER_PROXY" not in os.environ:
        print("No X509 proxy set. Exiting.")
        sys.exit(1)
    
    ## Check if we are in the cms-bot "environment"
    testing = "JENKINS_PREFIX" in os.environ
    dataset   = args.dataset
    events    = args.events
    threshold = args.threshold
    outfile   = args.outfile
    site      = args.site
    lumis     = args.lumis

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
    
    cert_path = base_cert_path + cert_type + "/"
    web_fallback = False

    ## if we have access to eos we get from there ...
    if os.path.isdir(cert_path):
        json_list = os.listdir(cert_path)
        if len(json_list) == 0:
            web_fallback == True
        json_list = [c for c in json_list if "Golden" in c and "era" not in c]
        json_list = [c for c in json_list if c.startswith("Cert_C") and c.endswith("json")]
    else:
        web_fallback = True
    ## ... if not we go to the website
    if web_fallback:
        cert_url = base_cert_url + cert_type + "/"
        json_list = get_url_clean(cert_url).split("\n")
        json_list = [c for c in json_list if "Golden" in c and "era" not in c and "Cert_C" in c]
        json_list = [[cc for cc in c.split(" ") if cc.startswith("Cert_C") and cc.endswith("json")][0] for c in json_list]

    # the larger the better, assuming file naming schema 
    # Cert_X_RunStart_RunFinish_Type.json
    run_ranges = [int(c.split("_")[3]) - int(c.split("_")[2]) for c in json_list]
    latest_json = np.array(json_list[np.argmax(run_ranges)]).reshape(1,-1)[0].astype(str)
    best_json = str(latest_json[0])
    if not web_fallback:
        with open(cert_path + "/" + best_json) as js:
            golden = json.load(js)
    else:
        golden = get_url_clean(cert_url + best_json)
        golden = ast.literal_eval(golden) #converts string to dict

    # golden json with all the lumisections
    golden_flat = {}
    for k in golden:
        R = []
        for r in golden[k]:
            R = R + [f for f in range(r[0],r[1]+1)]
        golden_flat[k] = R

    # let's just check there's an intersection between the
    # dataset and the json
    data_runs = das_run_data(dataset)
    golden_data_runs = [r for r in data_runs if r in golden_flat]

    if (len(golden_data_runs)==0):
        no_intersection()

    # building the dataframe, cleaning for bad lumis
    golden_data_runs_tocheck = golden_data_runs
    das_opt = ""
    if testing or args.precheck:
        golden_data_runs_tocheck = []
        # Here we check run per run.
        # This implies more dasgoclient queries, but smaller outputs
        # useful when running the IB/PR tests not to have huge
        # query results that have to be cached.

        sum_events = 0

        for r in golden_data_runs:
            sum_events = sum_events + int(das_run_events_data(dataset,r))
            golden_data_runs_tocheck.append(r)
            if events > 0 and sum_events > events:
                break

        das_opt = "run in %s"%(str([int(g) for g in golden_data_runs_tocheck]))

    df = das_lumi_data(dataset,opt=das_opt).merge(das_file_data(dataset,opt=das_opt),on="file",how="inner") # merge file informations with run and lumis

    df["lumis"] = [[int(ff) for ff in f.replace("[","").replace("]","").split(",")] for f in df.lumis.values]
    df_rs = []
    for r in golden_data_runs_tocheck:
        cut = (df["run"] == r)
        if not any(cut):
            continue

        df_r = df[cut]

        # jumping low event content runs
        if df_r["events"].sum() < threshold:
            continue

        good_lumis = np.array([len([ll for ll in l if ll in golden_flat[r]]) for l in df_r.lumis])
        n_lumis = np.array([len(l) for l in df_r.lumis])
        df_rs.append(df_r[good_lumis==n_lumis])

    if (len(df_rs)==0):
        no_intersection()

    df = pd.concat(df_rs)
    df.loc[:,"min_lumi"] = [min(f) for f in df.lumis]
    df.loc[:,"max_lumi"] = [max(f) for f in df.lumis]
    df = df.sort_values(["run","min_lumi","max_lumi"])

    if site is not None:
        df = df.merge(das_file_site(dataset,site),on="file",how="inner")

    if args.pandas:
        df.to_csv(dataset.replace("/","")+".csv")

    if events > 0:
        df = df[df["events"] <= events] #jump too big files
        df.loc[:,"sum_evs"] = df.loc[:,"events"].cumsum()
        df = df[df["sum_evs"] < events]
        
    files = df.file
    
    if lumis is not None:
        lumi_ranges = { int(r) : list(get_lumi_ranges(np.sort(np.concatenate(df.loc[df["run"]==r,"lumis"].values).ravel()).tolist())) for r in np.unique(df.run.values).tolist()}
        
        with open(lumis, 'w') as fp:
            json.dump(lumi_ranges, fp)

    if outfile is not None:
        with open(outfile, 'w') as f:
            for line in files:
                f.write(f"{line}\n") 
    else:
        print("\n".join(files))

    sys.exit(0)

    