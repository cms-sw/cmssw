#!/usr/bin/env python3
import json

def processModule(moduleLabel, moduleInfo, construction):
    for entry in moduleInfo:
        if entry["transition"] == "construction":
            construction[moduleLabel] = entry["alloc"]

def formatToCircles(construction):
    nevents = 1
    doc = {
        "modules": [],
        "resources": [
            {
                "name": "nAlloc",
                "description": "num allocs",
                "title": "Number of allocations",
                "unit": "B"
            },
            {
                "name": "added",
                "description": "added memory",
                "title": "Amount of memory added to the process at the end of the transition",
                "unit": "B"
            },
            {
                "name": "maxTemp",
                "description": "peak temporary memory",
                "title": "Peak temporary memory held during the transition",
                "unit": "B"
            }
        ],
        "total": {
            "events": nevents, # TODO
            "label": "FOO", # TODO
            "nAlloc": 1, # TODO
            "added": 1, # TODO
            "maxTemp": 1, # TODO
            "type": "Job"
        }
    }
    for label, info in construction.items():
        doc["modules"].append({
            "events" : nevents,
            "label": label,
            "type": "TODO", # TODO
            #"nAlloc": info["nAlloc"],
            "added": info["added"],
            #"maxTemp": info["maxTemp"],
        })
    return doc
            
def main(args):
    doc = json.load(args.filename)

    construction = dict()

    processModule("source", doc["source"], construction)
    for moduleLabel, moduleInfo in doc["modules"].items():
        processModule(moduleLabel, moduleInfo, construction)

    import sys
    json.dump(formatToCircles(construction), sys.stdout, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert the JSON output of edmModuleAllocMonitorAnalyze.py to JSON for Circles')
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')

    args = parser.parse_args()
    main(args)
