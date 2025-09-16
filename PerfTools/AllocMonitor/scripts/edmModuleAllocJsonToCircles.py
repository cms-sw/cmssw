#!/usr/bin/env python3
import json

def processModuleTransition(moduleLabel, moduleType, moduleInfo, moduleTransition):
    moduleTransition[moduleLabel] = {"cpptype": moduleType, "alloc": {}}
    for entry in moduleInfo:
        if entry["transition"] == "construction":
            moduleTransition[moduleLabel]["alloc"] = entry.get("alloc",{})

def formatToCircles(moduleTransition):
    nevents = 1
    doc = {
       "modules": [],
       "resources": [
           {
               "name": "added",
               "description": "added memory",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "Bytes"
           },
           {
               "name": "nAlloc",
               "description": "num allocs",
               "title": "Number of allocations",
               "unit": "count"
           },
           {
               "name": "max1Alloc",
               "description": "Maximum temporary allocations",
               "title": "Maximum temporary allocations held during the transition",
               "unit": "count"
           },
           {
               "name": "nDealloc",
               "description": "num deallocs",
               "title": "Number of deallocations",
               "unit": "count"
           },
           {
               "name": "minTemp",
               "description": "minimum temporary memory",
               "title": "Minimum temporary memory held during the transition",
               "unit": "Bytes"
           },
           {
               "name": "maxTemp",
               "description": "peak temporary memory",
               "title": "Maximum temporary memory held during the transition",
               "unit": "Bytes"
           }
       ],
       "total": {
           "events": 0,
           "label": "Job",
           "nAlloc": 0,
           "nDealloc": 0,
           "minTemp": 0,
           "max1Alloc": 0,
           "added": 0,
           "maxTemp": 0,
           "type": "Job"
       }
    }
    for label, info in moduleTransition.items():
        alloc=info.get("alloc", {})
        if alloc:
            added = alloc.get("added", 0)
            nAlloc = alloc.get("nAlloc", 0)
            nDealloc = alloc.get("nDealloc", 0)
            minTemp = alloc.get("minTemp", 0)
            maxTemp = alloc.get("maxTemp", 0)
            max1Alloc = alloc.get("max1Alloc", 0)
        doc["modules"].append({
            "events" : nevents,
            "label": label,
            "type": info.get("cpptype", "unknown"),
            "nAlloc": nAlloc,
            "added": added,
            "maxTemp": maxTemp,
            "nDealloc": nDealloc,
            "minTemp": minTemp,
            "max1Alloc": max1Alloc
        })
    return doc
            
def main(args):
    import sys
    doc = json.load(args.filename)
    moduleTypes = doc['cpptypes']
    if args.construction:
        moduleTransition = dict()
        processModuleTransition("source", "PoolSource", doc["source"], moduleTransition)
        for moduleLabel, moduleInfo in doc["modules"].items():
            processModuleTransition(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
        json.dump(formatToCircles(moduleTransition), sys.stdout, indent=2)
    if args.events:
        moduleTransition = dict()
        for moduleLabel, moduleInfo in doc["modules"].items():
            processModuleTransition(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
        json.dump(formatToCircles(moduleTransition), sys.stdout, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert the JSON output of edmModuleAllocMonitorAnalyze.py to JSON for Circles')
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    parser.add_argument('--construction',
                        action='store_true',
                        help='show construction transition memory use')
    parser.add_argument('--events',
                        action='store_true',
                        help='show event transition memory use')

    args = parser.parse_args()
    main(args)
