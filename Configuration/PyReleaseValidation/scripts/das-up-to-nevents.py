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

def das_do_command(cmd):
    out = subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode('utf8')
    return out

def das_file_data(dataset,opt=""):
    cmd = "dasgoclient --query='file dataset=%s %s| grep file.name, file.nevents'"%(dataset,opt)
    
    out = das_do_command(cmd).split("\n")
    out = [np.array(r.split(" "))[[0,3]] for r in out if len(r) > 0]
    
    df = pd.DataFrame(out,columns=["file","events"])
    df.events = df.events.values.astype(int)
    
    return df

def das_lumi_data(dataset,opt=""):
    cmd = "dasgoclient --query='file,lumi,run dataset=%s %s'"%(dataset,opt)
    
    out = das_do_command(cmd).split("\n")
    out = [r.split(" ") for r in out if len(r)>0]
    
    df = pd.DataFrame(out,columns=["file","run","lumis"])
    
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', default=None, help="Dataset Name (e.g. '/DisplacedJet/Run2024C-v1/RAW' )",type=str,required=True)
    parser.add_argument('--threshold','-t', help ="Event threshold per file",type=int,default=-1)
    parser.add_argument('--events','-e', help ="Tot number of events targeted",type=int,default=-1)
    parser.add_argument('--outfile','-o', help='Dump results to file', type=str, default=None)
    parser.add_argument('--pandas', '-pd',action='store_true',help="Store the whole dataset (no event or threshold cut) in a csv") 
    parser.add_argument('--proxy','-p', help='Allow to parse a x509 proxy if needed', type=str, default=None)
    parser.add_argument('--site','-s', help='Only data at specific site', type=str, default=None)
    args = parser.parse_args()

    if args.proxy is not None:
        os.environ["X509_USER_PROXY"] = args.proxy
    elif "X509_USER_PROXY" not in os.environ:
        print("No X509 proxy set. Exiting.")
        sys.exit(1)
    
    dataset   = args.dataset
    events    = args.events
    threshold = args.threshold
    outfile   = args.outfile
    site      = "site="+args.site if args.site is not None else ""

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

    if os.path.isdir(cert_path):
        json_list = os.listdir(cert_path)
        if len(json_list) == 0:
            web_fallback == True
        json_list = [c for c in json_list if "Golden" in c and "era" not in c]
        json_list = [c for c in json_list if c.startswith("Cert_C") and c.endswith("json")]
    else:
        web_fallback = True
    
    if web_fallback:
        cert_url = base_cert_url + cert_type + "/"
        json_list = get_url_clean(cert_url).split("\n")
        json_list = [c for c in json_list if "Golden" in c and "era" not in c]
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

    # building the dataframe, cleaning for bad lumis
    df = das_lumi_data(dataset).merge(das_file_data(dataset),on="file",how="inner") # merge file informations with run and lumis
    df = df[df["run"].isin(list(golden.keys()))] # skim for golden runs
    df["lumis"] = [[int(ff) for ff in f.replace("[","").replace("]","").split(",")] for f in df.lumis.values]
    df_rs = []
    for r in golden_flat:
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

    df = pd.concat(df_rs)
    df.loc[:,"min_lumi"] = [min(f) for f in df.lumis]
    df.loc[:,"max_lumi"] = [max(f) for f in df.lumis]
    df = df.sort_values(["run","min_lumi","max_lumi"])

    if args.pandas:
        df.to_csv(dataset.replace("/","")+".csv")

    if events > 0:
        df.loc[:,"sum_evs"] = df.loc[:,"events"].cumsum()
        df = df[df["sum_evs"] < events]
            
    files = df.file

    if outfile is not None:
        with open(outfile, 'w') as f:
            for line in files:
                f.write(f"{line}\n") 
    else:
        print("\n".join(files))

    sys.exit(0)

    
