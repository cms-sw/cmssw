#!/usr/bin/env python3
import json
import sys
from collections import namedtuple

# Constants
BYTES_TO_KB = 1024
EVENTSETUP_TRANSITION = "event setup"

# Named tuple for unique module identification
UniqueKey = namedtuple('UniqueKey', ['moduleLabel', 'moduleType', 'recordName'])

transitionTypes = [
    "construction",
    "destruction",
    "begin job",
    "begin stream",
    "get next transition",
    "global begin run",
    "stream begin run",
    "global begin luminosity block",
    "stream begin luminosity block",
    EVENTSETUP_TRANSITION,
    "event",
    "clear event",
    "stream end lumi",
    "global end lumi",
    "global write lumi",
    "stream end run",
    "global end run",
    "global write run",
    "write process block",
    "end stream",
    "end job",
]
allocTypes = ["added", "nAlloc", "nDealloc", "maxTemp", "max1Alloc"]

def processModuleTransition(moduleLabel, moduleType, moduleInfo, transitionType, moduleTransition):
    """
    Processes module transitions for a given transition type.

    The expected schema for each 'alloc' dictionary is:
        {
            "added": int,        # Bytes added during transition
            "nAlloc": int,       # Number of allocations
            "nDealloc": int,     # Number of deallocations
            "maxTemp": int,      # Maximum temporary memory (bytes)
            "max1Alloc": int     # Largest single allocation (bytes)
        }
    Any missing field defaults to 0.

    Note: Entries with record names are excluded as they belong to EventSetup transition only.

    For the "source" module and "event" transition, sum all alloc records with the same
    run, lumi, and event number.
    """
    moduleKey = UniqueKey(moduleLabel, moduleType, "")
    moduleTransition[moduleKey] = {"cpptype": moduleType, "allocs": []}

    # Special handling for "source" module with "event" transition
    if moduleLabel == "source" and transitionType == "event":
        # Group entries by (run, lumi, event)
        event_groups = {}
        for entry in moduleInfo:
            if (entry.get("transition", None) == transitionType and
                not ("record" in entry and "name" in entry["record"])):
                sync = entry.get("sync", {})
                key = (sync.get("run", 0), sync.get("lumi", 0), sync.get("event", 0))
                if key not in event_groups:
                    event_groups[key] = []
                event_groups[key].append(entry.get("alloc", {}))

        # Sum allocations for each event group
        for event_key, allocs in event_groups.items():
            summed_alloc = {
                "added": sum(a.get("added", 0) for a in allocs),
                "nAlloc": sum(a.get("nAlloc", 0) for a in allocs),
                "nDealloc": sum(a.get("nDealloc", 0) for a in allocs),
                "maxTemp": sum(a.get("maxTemp", 0) for a in allocs),
                "max1Alloc": sum(a.get("max1Alloc", 0) for a in allocs)
            }
            moduleTransition[moduleKey]["allocs"].append(summed_alloc)
    else:
        # Original processing for other modules/transitions
        for entry in moduleInfo:
            entryTransition = entry.get("transition", None)
            entryActivity = entry.get("activity", None)

            if entryTransition == transitionType:
                if (entryTransition == "event" and
                    entryActivity in ("acquire", "process") and
                    not ("record" in entry and "name" in entry["record"])):
                    moduleTransition[moduleKey]["allocs"].append(entry.get("alloc", {}))
                elif not ("record" in entry and "name" in entry["record"]):
                    moduleTransition[moduleKey]["allocs"].append(entry.get("alloc", {}))

    moduleTransition[moduleKey]["nTransitions"] = len(moduleTransition[moduleKey]["allocs"])

def processESModuleTransition(moduleLabel, moduleType, moduleInfo, moduleTransition):
    """Process EventSetup transitions - entries with record names

    Creates unique entries for each module+type+record combination.
    """
    # Group allocations by record name
    recordAllocations = {}
    for entry in moduleInfo:
        # EventSetup entries are those with a "record" field containing "name"
        if "record" in entry and "name" in entry["record"]:
            recordName = entry["record"]["name"]
            if recordName not in recordAllocations:
                recordAllocations[recordName] = []
            recordAllocations[recordName].append(entry.get("alloc", {}))

    # Create separate entries for each record
    for recordName, allocs in recordAllocations.items():
        # Create unique key: module + type + record
        uniqueKey = UniqueKey(moduleLabel, moduleType, recordName)
        moduleTransition[uniqueKey] = {
            "cpptype": moduleType,
            "allocs": allocs,
            "nTransitions": len(allocs),
            "moduleLabel": moduleLabel,
            "recordName": recordName
        }

def processExternalWorkTransition(moduleLabel, moduleType, moduleInfo, moduleTransition):
    """Process ExternalWork transitions - entries with acquire/produce activity

    Creates separate entries for each module+type+activity combination within the event transition.
    The recordName is set to 'acquire' or 'produce' based on the activity.
    """
    activityToRecord = {"acquire": "acquire", "process": "produce"}

    # Group allocations by activity
    activityAllocations = {}
    for entry in moduleInfo:
        if (entry.get("transition", None) == "event" and
            "record" not in entry):
            activity = entry.get("activity", "process")
            if activity not in activityAllocations:
                activityAllocations[activity] = []
            activityAllocations[activity].append(entry.get("alloc", {}))

    # Create separate entries for each activity
    for activity, allocs in activityAllocations.items():
        if activity in activityToRecord:
            recordName = activityToRecord[activity]
            uniqueKey = UniqueKey(moduleLabel, moduleType, recordName)
            moduleTransition[uniqueKey] = {
                "cpptype": moduleType,
                "allocs": allocs,
                "nTransitions": len(allocs),
                "moduleLabel": moduleLabel,
                "recordName": recordName
            }

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
               "title": f"{transitionType}: Amount of memory added to the process at the end of the transition",
               "unit": "kB"
            },
            {

               "name": f"nAlloc {transitionType}",
               "description": f"{transitionType}: num allocs (average)",
               "title": f"{transitionType}: Number of allocations during the transition",
               "unit": ""
            },
            {
               "name": f"nDealloc {transitionType}",
               "description": f"{transitionType}: num deallocs (average)",
               "title": f"{transitionType}: Number of deallocations during the transition",
               "unit": ""
            },
            {
               "name": f"maxTemp {transitionType}",
               "description": f"{transitionType}: maximum temporary memory (average)",
               "title": f"{transitionType}: Maximum temporary memory during the transition",
               "unit": "kB"
            },
            {
               "name": f"max1Alloc {transitionType}",
               "description": f"{transitionType}: largest single allocation (average)",
               "title": f"{transitionType}: Largest single allocation during the transition",
               "unit": "kB"
            },
        ]
    doc["total"]["label"] = "Job"
    doc["total"]["type"] = "Job"
    for transType in transitionTypes:
        for allocType in allocTypes:
            doc["total"][f"{allocType} {transType}"] = 0

    all_module_keys = set()
    for transitionType, moduleTransition in moduleTransitions.items():
        for uniqueKey in moduleTransition.keys():
            all_module_keys.add(uniqueKey)

    for displayKey in all_module_keys:
        if displayKey not in modules_dict:
            modules_dict[displayKey] = {
                "label": displayKey.moduleLabel,
                "type": displayKey.moduleType,
                "record": displayKey.recordName
            }

            for transType in transitionTypes:
                for allocType in allocTypes:
                    modules_dict[displayKey][f"{allocType} {transType}"] = 0.0

    # Initialize acquire/produce totals
    for displayKey in all_module_keys:
        if displayKey.recordName in ("acquire", "produce"):
            for allocType in allocTypes:
                doc["total"][f"{allocType} {displayKey.recordName}"] = 0

    for transitionType, moduleTransition in moduleTransitions.items():
        for uniqueKey, info in moduleTransition.items():
            allocs = info.get("allocs", [])

            if uniqueKey in modules_dict:
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
                ntransitions = moduleTransitions[transitionType][uniqueKey].get("nTransitions", -1)
                divisor = max(ntransitions, 1)

                metricSuffix = transitionType
                if uniqueKey.recordName in ("acquire", "produce"):
                    metricSuffix = uniqueKey.recordName
                elif transitionType == EVENTSETUP_TRANSITION and uniqueKey.recordName:
                    metricSuffix = uniqueKey.recordName

                modules_dict[uniqueKey][f"nAlloc {metricSuffix}"] = nAlloc / divisor
                modules_dict[uniqueKey][f"nDealloc {metricSuffix}"] = nDealloc / divisor
                modules_dict[uniqueKey][f"added {metricSuffix}"] = (added / divisor) / BYTES_TO_KB
                modules_dict[uniqueKey][f"maxTemp {metricSuffix}"] = (maxTemp / divisor) / BYTES_TO_KB
                modules_dict[uniqueKey][f"max1Alloc {metricSuffix}"] = (max1Alloc / divisor) / BYTES_TO_KB
                doc["total"][f"nAlloc {metricSuffix}"] += modules_dict[uniqueKey][f"nAlloc {metricSuffix}"]
                doc["total"][f"nDealloc {metricSuffix}"] += modules_dict[uniqueKey][f"nDealloc {metricSuffix}"]
                doc["total"][f"maxTemp {metricSuffix}"] += modules_dict[uniqueKey][f"maxTemp {metricSuffix}"]
                doc["total"][f"added {metricSuffix}"] += modules_dict[uniqueKey][f"added {metricSuffix}"]
                doc["total"][f"max1Alloc {metricSuffix}"] += modules_dict[uniqueKey][f"max1Alloc {metricSuffix}"]

    for key in sorted(modules_dict.keys()):
        module = modules_dict[key]

        if module["record"] == "":
            hasNonZeroAllocations = False
            for transType in transitionTypes:
                for allocType in allocTypes:
                    if module.get(f"{allocType} {transType}", 0) != 0:
                        hasNonZeroAllocations = True
                        break
                if hasNonZeroAllocations:
                    break

            if not hasNonZeroAllocations:
                continue

        moduleLabel = key.moduleLabel
        moduleTypeVal = key.moduleType
        recordName = key.recordName

        # For ExternalWork modules with acquire/produce record, use that record name for event count
        if recordName in ("acquire", "produce"):
            eventKey = UniqueKey(moduleLabel, moduleTypeVal, recordName)
            eventCount = moduleTransitions['event'].get(eventKey, {}).get("nTransitions", 0)
        else:
            eventKey = UniqueKey(moduleLabel, moduleTypeVal, "")
            eventCount = moduleTransitions['event'].get(eventKey, {}).get("nTransitions", 0)

        module["transitions"] = max(eventCount, 1)
        doc["modules"].append(module)

    return doc

def main(args):
    try:
        doc = json.load(args.filename)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if 'cpptypes' not in doc:
        print("Error: Missing 'cpptypes' field in input JSON", file=sys.stderr)
        sys.exit(1)
    if 'modules' not in doc:
        print("Error: Missing 'modules' field in input JSON", file=sys.stderr)
        sys.exit(1)

    moduleTypes = doc['cpptypes']
    moduleTransitions = dict()
    externalWorkModules = set()

    for transition in transitionTypes:
        moduleTransition = dict()
        if transition == EVENTSETUP_TRANSITION:
            for moduleLabel, moduleInfo in doc["modules"].items():
                processESModuleTransition(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
        else:
            processModuleTransition("source", moduleTypes["source"], doc["source"], transition, moduleTransition)
            for moduleLabel, moduleInfo in doc["modules"].items():
                moduleType = moduleTypes[moduleLabel]
                if "ExternalWork" in moduleType or "Transform" in moduleType:
                    externalWorkModules.add(moduleLabel)
                processModuleTransition(moduleLabel, moduleType, moduleInfo, transition, moduleTransition)
        moduleTransitions[transition] = moduleTransition

    for moduleLabel in externalWorkModules:
        moduleTransition = dict()
        processExternalWorkTransition(moduleLabel, moduleTypes[moduleLabel], doc["modules"][moduleLabel], moduleTransition)
        for uniqueKey, info in moduleTransition.items():
            moduleTransitions['event'][uniqueKey] = info

    json.dump(formatToCircles(moduleTransitions), sys.stdout, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert the JSON output of edmModuleAllocMonitorAnalyze.py to JSON for Circles')
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    args = parser.parse_args()
    main(args)
