#!/usr/bin/env python3
import json
transitionTypes = ["construction", "event",
                        "begin job", "begin stream",
                        "global begin run", "global begin luminosity block",
                        "stream begin run", "stream begin luminosity block",
                        ]

def processModuleTransition(moduleLabel, moduleType, moduleInfo, transitionType, moduleTransition):
        moduleTransition[moduleLabel] = {"cpptype": moduleType, "allocs": [], "nEvents": 0}
        for entry in moduleInfo:
            if entry["transition"] == transitionType:
                moduleTransition[moduleLabel]["allocs"].append(entry.get("alloc",{}))
        moduleTransition[moduleLabel]["nEvents"] = len(moduleTransition[moduleLabel]["allocs"])

def formatToCircles(moduleTransitions):
    nevents = 1
    modules_dict = {}
    doc = {
       "modules": [],
       "resources": [
            {
               "name": "added event",
               "description": "added memory average per event transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc event",
               "description": "num allocs average per event transition",
               "title": "Number of allocations",
               "unit": ""
            },

           {
               "name": "max1Alloc event",
               "description": "maximum temporary allocations average per event transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc event",
               "description": "num deallocs average per event",
               "title": "Number of deallocations per event",
               "unit": ""
           },
           {
               "name": "maxTemp event",
               "description": "peak temporary memory average per event transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added construction",
               "description": "added memory construction transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc construction",
               "description": "num allocs for construction transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc construction",
               "description": "maximum temporary allocations average for construction transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc construction",
               "description": "num deallocs for construction transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp construction",
               "description": "peak temporary memory for construction transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added begin job",
               "description": "added memory begin job transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc begin job",
               "description": "num allocs for begin job transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc begin job",
               "description": "maximum temporary allocations average for begin job transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc begin job",
               "description": "num deallocs for begin job transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp begin job",
               "description": "peak temporary memory for begin job transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added begin stream",
               "description": "added memory begin stream transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc begin stream",
               "description": "num allocs for begin stream transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc begin stream",
               "description": "maximum temporary allocations average for begin stream transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc begin stream",
               "description": "num deallocs for begin stream transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp begin stream",
               "description": "peak temporary memory for begin stream transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added global begin run",
               "description": "added memory global begin run transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc global begin run",
               "description": "num allocs for global begin run transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc global begin run",
               "description": "maximum temporary allocations average for global begin run transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc global begin run",
               "description": "num deallocs for global begin run transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp global begin run",
               "description": "peak temporary memory for global begin run transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added global begin luminosity block",
               "description": "added memory global begin luminosity block transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc global begin luminosity block",
               "description": "num allocs for global begin luminosity block transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc global begin luminosity block",
               "description": "maximum temporary allocations average for global begin luminosity block transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc global begin luminosity block",
               "description": "num deallocs for global begin luminosity block transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp global begin luminosity block",
               "description": "peak temporary memory for global begin luminosity block transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added stream begin run",
               "description": "added memory stream begin run transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc stream begin run",
               "description": "num allocs for stream begin run transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc stream begin run",
               "description": "maximum temporary allocations average for stream begin run transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc stream begin run",
               "description": "num deallocs for stream begin run transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp stream begin run",
               "description": "peak temporary memory for stream begin run transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
           {
               "name": "added stream begin luminosity block",
               "description": "added memory stream begin luminosity block transition",
               "title": "Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": "nAlloc stream begin luminosity block",
               "description": "num allocs for stream begin luminosity block transition",
               "title": "Number of allocations",
               "unit": ""
           },
           {
               "name": "max1Alloc stream begin luminosity block",
               "description": "maximum temporary allocations average for stream begin luminosity block transition",
               "title": "Maximum temporary allocations held during the transition",
               "unit": ""
           },
           {
               "name": "nDealloc stream begin luminosity block",
               "description": "num deallocs for stream begin luminosity block transition",
               "title": "Number of deallocations",
               "unit": ""
           },
           {
               "name": "maxTemp stream begin luminosity block",
               "description": "peak temporary memory for stream begin luminosity block transition",
               "title": "Maximum temporary memory held during the transition",
               "unit": "kB"
           },
       ],
       "total": {
           "events": 1,
           "label": "Job",
           "type": "Job",
           "nAlloc event": 0,
           "nDealloc event": 0,
           "max1Alloc event": 0,
           "added event": 0,
           "maxTemp event": 0,
           "nAlloc construction": 0,
           "nDealloc construction": 0,
           "max1Alloc construction": 0,
           "added construction": 0,
           "maxTemp construction": 0,
           "nAlloc begin job": 0,
           "nDealloc begin job": 0,
           "max1Alloc begin job": 0,
           "added begin job": 0,
           "maxTemp begin job": 0,
           "nAlloc begin stream": 0,
           "nDealloc begin stream": 0,
           "max1Alloc begin stream": 0,
           "added begin stream": 0,
           "maxTemp begin stream": 0,
           "nAlloc global begin run": 0,
           "nDealloc global begin run": 0,
           "max1Alloc global begin run": 0,
           "added global begin run": 0,
           "maxTemp global begin run": 0,
           "nAlloc global begin luminosity block": 0,
           "nDealloc global begin luminosity block": 0,
           "max1Alloc global begin luminosity block": 0,
           "added global begin luminosity block": 0,
           "maxTemp global begin luminosity block": 0,
           "nAlloc stream begin run": 0,
           "nDealloc stream begin run": 0,
           "max1Alloc stream begin run": 0,
           "added stream begin run": 0,
           "maxTemp stream begin run": 0,
           "nAlloc stream begin luminosity block": 0,
           "nDealloc stream begin luminosity block": 0,
           "max1Alloc stream begin luminosity block": 0,
           "added stream begin luminosity block": 0,
           "maxTemp stream begin luminosity block": 0,
       }
    }
    for transitionType in moduleTransitions.keys():
        for label, info in moduleTransitions[transitionType].items():
            allocs = info.get("allocs", [])
            modules_dict[label] = modules_dict.get(label, {})
            modules_dict[label]["label"] = info.get("label", label)
            modules_dict[label]["type"] = info.get("cpptype", "unknown")
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
            nevents = moduleTransitions[transitionType][label]["nEvents"]
            if nevents > 0:
                modules_dict[label]["nAlloc %s" % transitionType] = nAlloc/nevents
                modules_dict[label]["added %s" % transitionType] = (added/nevents)/1024
                modules_dict[label]["maxTemp %s" % transitionType] = (maxTemp/nevents)/1024
                modules_dict[label]["nDealloc %s" % transitionType] = nDealloc/nevents
                modules_dict[label]["max1Alloc %s" % transitionType] = max1Alloc/nevents
            else:
                modules_dict[label]["nAlloc %s" % transitionType] = nAlloc
                modules_dict[label]["added %s" % transitionType] = (added)/1024
                modules_dict[label]["maxTemp %s" % transitionType] = (maxTemp)/1024
                modules_dict[label]["nDealloc %s" % transitionType] = nDealloc
                modules_dict[label]["max1Alloc %s" % transitionType] = max1Alloc
            doc["total"]["nAlloc %s" % transitionType] += nAlloc
            doc["total"]["nDealloc %s" % transitionType] += nDealloc
            doc["total"]["maxTemp %s" % transitionType] += maxTemp
            doc["total"]["added %s" % transitionType] += added
            doc["total"]["max1Alloc %s" % transitionType] += max1Alloc
            doc["total"]["events"] = moduleTransitions['event'][label].get("nEvents", 1)
    for key in sorted(modules_dict.keys()):
        module = modules_dict[key]
        module["events"] = moduleTransitions['event'][key].get("nEvents", 1)
        if module["events"] == 0:
            module["events"] = 1
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
