#!/usr/bin/env python3
import json

def processModuleConstruction(moduleLabel, moduleType, moduleInfo, moduleTransition):
    moduleTransition[moduleLabel] = {"cpptype": moduleType, "allocs": [], "nEvents": 0}
    for entry in moduleInfo:
        if entry["transition"] == "construction":
            moduleTransition[moduleLabel]["allocs"].append(entry.get("alloc",{}))
        moduleTransition[moduleLabel]["nEvents"] = len(moduleTransition[moduleLabel]["allocs"])
def processModuleEvents(moduleLabel, moduleType, moduleInfo, moduleTransition):
    moduleTransition[moduleLabel] = {"cpptype": moduleType, "allocs": [], "nEvents": 0}
    for entry in moduleInfo:
        if entry["transition"] == "event":
            moduleTransition[moduleLabel]["allocs"].append(entry.get("alloc",{}))
        moduleTransition[moduleLabel]["nEvents"] = len(moduleTransition[moduleLabel]["allocs"])

def formatToCircles(moduleTransition):
    nevents = 1
    doc = {
       "modules": [],
       "resources": [
           {
               "name": "added",
               "description": "added memory",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
           },
           {
               "name": "nAlloc",
               "description": "num allocs",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc",
               "description": "Maximum temporary allocations",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc",
               "description": "num deallocs",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "minTemp",
               "description": "minimum temporary memory",
               "title": "Minimum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "maxTemp",
               "description": "peak temporary memory",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           }
       ],
       "total": {
           "events": 1,
           "label": "Job",
           "nAlloc": 1,
           "nDealloc": 1,
           "minTemp": 1,
           "max1Alloc": 1,
           "added": 1,
           "maxTemp": 1,
           "type": "Job"
       }
    }
    for label, info in moduleTransition.items():
        allocs = info.get("allocs", [])
        added = 0
        nAlloc = 0
        nDealloc = 0
        minTemp = 0
        maxTemp = 0
        max1Alloc = 0
        nevents = moduleTransition[label]["nEvents"]
        for alloc in allocs:
            added += alloc.get("added", 0)
            nAlloc += alloc.get("nAlloc", 0)
            nDealloc += alloc.get("nDealloc", 0)
            minTemp += alloc.get("minTemp", 0)
            maxTemp += alloc.get("maxTemp", 0)
            max1Alloc += alloc.get("max1Alloc", 0)
        if nevents > 0:
            doc["modules"].append({
                "events" : nevents,
                "label": label,
                "type": info.get("cpptype", "unknown"),
                "nAlloc": nAlloc/nevents,
                "added": (added/nevents)/1024,
                "maxTemp": (maxTemp/nevents)/1024,
                "nDealloc": nDealloc/nevents,
                "minTemp": (minTemp/nevents)/1024,
                "max1Alloc": max1Alloc
            })
        doc["total"]["nAlloc"] += nAlloc
        doc["total"]["nDealloc"] += nDealloc
        doc["total"]["minTemp"] += minTemp
        doc["total"]["maxTemp"] += maxTemp
        doc["total"]["added"] += added
        doc["total"]["max1Alloc"] += max1Alloc
        doc["total"]["events"] = moduleTransition["source"]["nEvents"]
    return doc
            
def main(args):
    import sys
    doc = json.load(args.filename)
    moduleTypes = doc['cpptypes']
    if args.construction:
        moduleTransition = dict()
        processModuleConstruction("source", "PoolSource", doc["source"], moduleTransition)
        for moduleLabel, moduleInfo in doc["modules"].items():
            processModuleConstruction(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
        json.dump(formatToCircles(moduleTransition), sys.stdout, indent=2)
    if args.events:
        moduleTransition = dict()
        processModuleEvents("source", "PoolSource", doc["source"], moduleTransition)
        for moduleLabel, moduleInfo in doc["modules"].items():
            processModuleEvents(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
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
