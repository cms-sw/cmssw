#!/usr/bin/env python3
import json
import requests
import argparse
import subprocess

BASEURL = None
XRDPREFIX = "root://cms-xrd-global.cern.ch/"
VERBOSE = False

def dasquery(query, usejson=False):
    q = ['dasgoclient', '-query', query] + (['--json'] if usejson else [])
    if VERBOSE:
        print(q)
    datasets = subprocess.check_output(q)
    if usejson:
        return json.loads(datasets.decode("utf-8"))
    else:
        return [s.decode("utf-8") for s in datasets.splitlines()]

def findreadabledatasets(pattern):
    out = []
    datasets = dasquery(f'dataset dataset={pattern}')
    print(f"Got {len(datasets)} datasets matching {pattern}.")
    for d in datasets:
        sites = dasquery(f'site dataset={d}', usejson=True)
        for site in sites:
            siteinfo = site['site']
            for thing in siteinfo:
                #print (thing)
                if 'kind' in thing and thing['kind'] == 'Disk':
                    print(f"Dataset {d} on disk as {thing['name']}.")
                    out.append(d)
    return out

def querysamples(dataset):
    lumiinfo = dasquery(f"file run lumi dataset={dataset}")
    samples = []
    for row in lumiinfo:
        parts = row.split(" ")
        file = parts[0]
        run = parts[1]
        lumis = parts[2][1:-1].split(",")
        for lumi in lumis:
            samples.append((file, run, lumi))
    print(f"Got {len(samples)} for dataset {dataset}.")
    return samples

def registersamples(dataset, samples):
    chuncksize = 1000
    for i in range(0, len(samples), chuncksize):
        postdata = json.dumps(
            [{"dataset":dataset , "run": run, "lumi": lumi, "file": XRDPREFIX + file, "fileformat": 2}
                for file, run, lumi in samples[i:i+chuncksize]]
        )
        res = requests.post(BASEURL+"/api/v1/register", data=postdata)
        print(f"Registerd samples from {i} of {dataset}, response {res}.")
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQMIO reagistration helper. Queries DAS for DQMIO data and registers it.')
    parser.add_argument('--server', default="http://localhost:8889", help='DQMGUI server to register to.')
    parser.add_argument('--dataset', default="/ZeroBias/*UL2018*/DQMIO" , help='Dataset pattern for DAS.')
    parser.add_argument('--verbose', action="store_true" , help='Print all queries.')
    args = parser.parse_args()
    BASEURL = args.server
    VERBOSE = args.verbose

    datasets = findreadabledatasets(args.dataset)
    for dataset in datasets:
        registersamples(dataset, querysamples(dataset))





