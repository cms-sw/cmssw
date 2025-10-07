#!/usr/bin/env python3
import json
import sys

# Constants
BYTES_TO_KB = 1024

transitionTypes = [
    "construction",
    "begin job",
    "begin stream",
    "global begin run",
    "stream begin run",
    "global begin luminosity block",
    "stream begin luminosity block",
    "event",
    "event setup",
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

    Note: Entries with record names are excluded as they belong to event setup transition only.
    """
    moduleTransition[moduleLabel] = {"cpptype": moduleType, "allocs": []}
    for entry in moduleInfo:
        # Only process entries that match the transition type AND don't have record names
        # (entries with record names are event setup-only)
        if (entry.get("transition", None) == transitionType and
            not ("record" in entry and "name" in entry["record"])):
            moduleTransition[moduleLabel]["allocs"].append(entry.get("alloc", {}))
    moduleTransition[moduleLabel]["nTransitions"] = len(moduleTransition[moduleLabel]["allocs"])

def processESProducerTransition(moduleLabel, moduleType, moduleInfo, moduleTransition):
    """Process event setup transitions - entries with record names

    Creates unique entries for each module+type+record combination.
    """
    # Group allocations by record name
    recordAllocations = {}
    for entry in moduleInfo:
        # event setup entries are those with a "record" field containing "name"
        if "record" in entry and "name" in entry["record"]:
            recordName = entry["record"]["name"]
            if recordName not in recordAllocations:
                recordAllocations[recordName] = []
            recordAllocations[recordName].append(entry.get("alloc", {}))

    # Create separate entries for each record
    for recordName, allocs in recordAllocations.items():
        # Create unique key: module + type + record
        uniqueKey = f"{moduleLabel}::{moduleType}::{recordName}"
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
    # The circles code uses the "events" field to normalize the values between files with different number of events
    # Here we set it to 1 for the total events because the total is already normalized per transition
    doc["total"]["events"] = 1
    doc["total"]["label"] = "Job"
    doc["total"]["type"] = "Job"
    # Initialize totals for all transition types and allocation types
    for transType in transitionTypes:
        for allocType in allocTypes:
            doc["total"][f"{allocType} {transType}"] = 0

    # First pass: collect all unique module keys across all transitions
    all_module_keys = set()
    for transitionType, moduleTransition in moduleTransitions.items():
        for uniqueKey in moduleTransition.keys():
            if transitionType == "event setup":
                displayKey = uniqueKey
            else:
                displayKey = uniqueKey  # For regular transitions, this is just the module label
            all_module_keys.add(displayKey)

    # Initialize all modules with default values for all transitions
    for displayKey in all_module_keys:
        if displayKey not in modules_dict:
            # Determine module info from the key
            if "::" in displayKey:
                # event setup key format: moduleLabel::moduleType::recordName
                parts = displayKey.split("::", 2)
                moduleLabel = parts[0]
                moduleType = parts[1]
                recordName = parts[2]
            else:
                # Regular module key
                moduleLabel = displayKey
                # Find the module type from any transition
                moduleType = "unknown"
                for transType, moduleTransition in moduleTransitions.items():
                    if displayKey in moduleTransition:
                        moduleType = moduleTransition[displayKey].get("cpptype", "unknown")
                        break
                recordName = ""

            modules_dict[displayKey] = {
                "label": moduleLabel,
                "type": moduleType,
                "record": recordName
            }

            # Initialize all transition metrics to zero
            for transType in transitionTypes:
                for allocType in allocTypes:
                    modules_dict[displayKey][f"{allocType} {transType}"] = 0.0

    # Second pass: populate actual values
    for transitionType, moduleTransition in moduleTransitions.items():
        for uniqueKey, info in moduleTransition.items():
            allocs = info.get("allocs", [])

            # For event setup transitions, use the unique key; for others, use original label
            if transitionType == "event setup":
                displayKey = uniqueKey
            else:
                displayKey = uniqueKey  # For regular transitions, this is just the module label
            # Only update metrics if this module actually has data for this transition
            if displayKey in modules_dict:
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
                ntransitions = moduleTransitions[transitionType][uniqueKey].get("nTransitions", 0)
                # Normalize by number of transitions if > 0, otherwise use raw values
                divisor = max(ntransitions, 1)  # Avoid division by zero

                modules_dict[displayKey][f"nAlloc {transitionType}"] = nAlloc / divisor
                modules_dict[displayKey][f"nDealloc {transitionType}"] = nDealloc / divisor
                modules_dict[displayKey][f"added {transitionType}"] = (added / divisor) / BYTES_TO_KB
                modules_dict[displayKey][f"maxTemp {transitionType}"] = (maxTemp / divisor) / BYTES_TO_KB
                modules_dict[displayKey][f"max1Alloc {transitionType}"] = (max1Alloc / divisor) / BYTES_TO_KB
                doc["total"][f"nAlloc {transitionType}"] += modules_dict[displayKey][f"nAlloc {transitionType}"]
                doc["total"][f"nDealloc {transitionType}"] += modules_dict[displayKey][f"nDealloc {transitionType}"]
                doc["total"][f"maxTemp {transitionType}"] += modules_dict[displayKey][f"maxTemp {transitionType}"]
                doc["total"][f"added {transitionType}"] += modules_dict[displayKey][f"added {transitionType}"]
                doc["total"][f"max1Alloc {transitionType}"] += modules_dict[displayKey][f"max1Alloc {transitionType}"]

    for key in sorted(modules_dict.keys()):
        module = modules_dict[key]
 
        # Check if this is an empty entry (record="" and all allocations are zero)
        if module["record"] == "":
            # Check if all allocation metrics are zero across all transition types
            hasNonZeroAllocations = False
            for transType in transitionTypes:
                for allocType in allocTypes:
                    if module.get(f"{allocType} {transType}", 0) != 0:
                        hasNonZeroAllocations = True
                        break
                if hasNonZeroAllocations:
                    break

            # Skip this entry if no allocations and empty record
            if not hasNonZeroAllocations:
                continue

        # For event setup entries (with ::), use the module label part for events count
        # For regular entries, use the key directly
        if "::" in key:
            moduleLabel = key.split("::")[0]
        else:
            moduleLabel = key
        eventCount = moduleTransitions['event'].get(moduleLabel, {}).get("nTransitions", 0)
        # Set events to 1 if it's 0 to prevent NaNs in Circles visualization
        module["events"] = max(eventCount, 1)
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

    # Validate required fields
    if 'cpptypes' not in doc:
        print("Error: Missing 'cpptypes' field in input JSON", file=sys.stderr)
        sys.exit(1)
    if 'modules' not in doc:
        print("Error: Missing 'modules' field in input JSON", file=sys.stderr)
        sys.exit(1)

    moduleTypes = doc['cpptypes']
    moduleTransitions = dict()
    for transition in transitionTypes:
        moduleTransition = dict()
        if transition == "event setup":
            # event setup transitions are handled differently - look for records with names
            processESProducerTransition("source", "PoolSource", doc["source"], moduleTransition)
            for moduleLabel, moduleInfo in doc["modules"].items():
                processESProducerTransition(moduleLabel, moduleTypes[moduleLabel], moduleInfo, moduleTransition)
        else:
            # Regular transition processing
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
