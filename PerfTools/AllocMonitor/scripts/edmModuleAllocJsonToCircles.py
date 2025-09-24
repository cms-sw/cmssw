#!/usr/bin/env python3
import json
transitionTypes = ["construction", "event",
                        "begin job", "begin stream",
                        "global begin run", "global begin luminosity block",
                        "stream begin run", "stream begin luminosity block",
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
               "name": "added %s" % transitionType,
               "description": "added memory per %s transition" % transitionType,
               "title": "Amount of memory added to the process at the end of the %s transition" % transitionType,
               "unit": "kB"
            },
            {

               "name": "nAlloc %s" % transitionType,
               "description": "num allocs per %s transition" % transitionType,
               "title": "Number of allocations during the %s transition" % transitionType,
               "unit": ""
            },
            {
               "name": "max1Alloc %s" % transitionType,
               "description": "maximum one time allocation per %s transition" % transitionType,
               "title": "Maximum one time allocation held during the %s transition" % transitionType,
               "unit": "kB"
           },
           {
               "name": "nDealloc %s" % transitionType,
               "description": "num deallocs for %s transition" % transitionType,
               "title": "Number of deallocations during the %s transition" % transitionType,
               "unit": ""
           },
           {
               "name": "maxTemp %s" % transitionType,
               "description": "maximum temporary memory per %s transition" % transitionType,
               "title": "Maximum temporary memory held during the %s transition" % transitionType,
               "unit": "kB"
           },
        ]
# The circles code uses the "events" field to normalize the values between files with different number of events
# Here we set it to 1 for the total events because the total is already normalized per transition
        doc["total"]["events"] = 1
        doc["total"]["label"] = "Job"
        doc["total"]["type"] = "Job"
        for allocType in allocTypes:
            doc["total"]["%s %s" % (allocType, transitionType)] = 0

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
                modules_dict[label]["nAlloc %s" % transitionType] = nAlloc/ntransitions
                modules_dict[label]["added %s" % transitionType] = (added/ntransitions)/1024
                modules_dict[label]["maxTemp %s" % transitionType] = (maxTemp/ntransitions)/1024
                modules_dict[label]["nDealloc %s" % transitionType] = nDealloc/ntransitions
                modules_dict[label]["max1Alloc %s" % transitionType] = (max1Alloc/ntransitions)/1024
            else:
                modules_dict[label]["nAlloc %s" % transitionType] = nAlloc
                modules_dict[label]["added %s" % transitionType] = (added)/1024
                modules_dict[label]["maxTemp %s" % transitionType] = (maxTemp)/1024
                modules_dict[label]["nDealloc %s" % transitionType] = nDealloc
                modules_dict[label]["max1Alloc %s" % transitionType] = max1Alloc/1024
            doc["total"]["nAlloc %s" % transitionType] += modules_dict[label]["nAlloc %s" % transitionType] 
            doc["total"]["nDealloc %s" % transitionType] += modules_dict[label]["nDealloc %s" % transitionType]
            doc["total"]["maxTemp %s" % transitionType] += modules_dict[label]["maxTemp %s" % transitionType]
            doc["total"]["added %s" % transitionType] += modules_dict[label]["added %s" % transitionType]
            doc["total"]["max1Alloc %s" % transitionType] += modules_dict[label]["max1Alloc %s" % transitionType]

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
