#!/usr/bin/env python3
import json
transitionTypes = [
    "construction",
    "begin job",
    "begin stream",
    "global begin run",
    "stream begin run",
    "global begin luminosity block",
    "stream begin luminosity block",
    "event",
]
allocTypes = ["added", "nAlloc", "nDealloc", "maxTemp", "max1Alloc"]

def processModuleTransition(moduleLabel, moduleType, moduleInfo, transitionType, moduleTransition):
        moduleTransition[moduleLabel] = {"cpptype": moduleType, "allocs": []}
        for entry in moduleInfo:
            if entry["transition"] == transitionType:
                moduleTransition[moduleLabel]["allocs"].append(entry.get("alloc",{}))
        moduleTransition[moduleLabel]["nTransitions"] = len(moduleTransition[moduleLabel]["allocs"])

def formatToCircles(moduleTransitions):
    modules_dict = {}
    doc = {
       "modules": [],
       "resources": [],
       "total": {}
    }
    for transitionType in transitionTypes:
        doc["resources"] += [
            {
               "name": f"added {transitionType}",
               "description": f"{transitionType}: added memory (average)",
               "title": f"Amount of memory added to the process at the end of the {transitionType} transition",
               "unit": "kB"
            },
            {

               "name": f"nAlloc {transitionType}",
               "description": f"{transitionType}: num allocs (average)",
               "title": f"Number of allocations during the {transitionType} transition",
               "unit": ""
            },
            {
               "name": f"nDealloc {transitionType}",
               "description": f"{transitionType}: num deallocs (average)",
               "title": f"Number of deallocations during the {transitionType} transition",
               "unit": ""
            },
            {
               "name": f"maxTemp {transitionType}",
               "description": f"{transitionType}: maximum temporary memory (average)",
               "title": f"Maximum temporary memory held during the {transitionType} transition",
               "unit": "kB"
            },
            {
               "name": f"max1Alloc {transitionType}",
               "description": f"{transitionType}: largest single allocation (average)",
               "title": f"Largest single allocation held during the {transitionType} transition",
               "unit": "kB"
            },
        ]
    # The circles code uses the "events" field to normalize the values between files with different number of events
    # Here we set it to 1 for the total events because the total is already normalized per transition
        doc["total"]["events"] = 1
        doc["total"]["label"] = "Job"
        doc["total"]["type"] = "Job"
        for allocType in allocTypes:
            doc["total"][f"{allocType} {transitionType}"] = 0

    for transitionType, moduleTransition in moduleTransitions.items():
        for label, info in moduleTransition.items():
            allocs = info.get("allocs", [])
            if not label in modules_dict:
                modules_dict[label] = {
                    "label": info.get("label", label),
                    "type": info.get("cpptype", "unknown")
                }
            added = 0
            nAlloc = 0
            nDealloc = 0
            maxTemp = 0
            max1Alloc = 0
            for alloc in allocs:
                added += alloc.get("added", 0)
                nAlloc += alloc.get("nAlloc", 0)
                nDealloc += alloc.get("nDealloc", 0)
                maxTemp += alloc.get("maxTemp", 0)
                max1Alloc += alloc.get("max1Alloc", 0)
            ntransitions = moduleTransitions[transitionType][label]["nTransitions"]
            if ntransitions > 0:
                modules_dict[label][f"nAlloc {transitionType}"] = nAlloc/ntransitions
                modules_dict[label][f"added {transitionType}"] = (added/ntransitions)/1024
                modules_dict[label][f"maxTemp {transitionType}"] = (maxTemp/ntransitions)/1024
                modules_dict[label][f"nDealloc {transitionType}"] = nDealloc/ntransitions
                modules_dict[label][f"max1Alloc {transitionType}"] = (max1Alloc/ntransitions)/1024
            else:
                modules_dict[label][f"nAlloc {transitionType}"] = nAlloc
                modules_dict[label][f"added {transitionType}"] = (added)/1024
                modules_dict[label][f"maxTemp {transitionType}"] = (maxTemp)/1024
                modules_dict[label][f"nDealloc {transitionType}"] = nDealloc
                modules_dict[label][f"max1Alloc {transitionType}"] = max1Alloc/1024
            doc["total"][f"nAlloc {transitionType}"] += modules_dict[label][f"nAlloc {transitionType}"]
            doc["total"][f"nDealloc {transitionType}"] += modules_dict[label][f"nDealloc {transitionType}"]
            doc["total"][f"maxTemp {transitionType}"] += modules_dict[label][f"maxTemp {transitionType}"]
            doc["total"][f"added {transitionType}"] += modules_dict[label][f"added {transitionType}"]
            doc["total"][f"max1Alloc {transitionType}"] += modules_dict[label][f"max1Alloc {transitionType}"]

    for key in sorted(modules_dict.keys()):
        module = modules_dict[key]
        module["events"] = moduleTransitions['event'][key].get("nTransitions")
        doc["modules"].append(module)

    return doc

def main(args):
    import sys
    doc = json.load(args.filename)
    moduleTypes = doc['cpptypes']
    moduleTransitions = dict()
    for transition in transitionTypes:
        moduleTransition = dict()
        processModuleTransition("source", "PoolSource", doc["source"], transition, moduleTransition)
        for moduleLabel, moduleInfo in doc["modules"].items():
            processModuleTransition(moduleLabel, moduleTypes[moduleLabel], moduleInfo, transition, moduleTransition)
        moduleTransitions[transition] = moduleTransition

    json.dump(formatToCircles(moduleTransitions), sys.stdout, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert the JSON output of edmModuleAllocMonitorAnalyze.py to JSON for Circles')
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    args = parser.parse_args()
    main(args)
