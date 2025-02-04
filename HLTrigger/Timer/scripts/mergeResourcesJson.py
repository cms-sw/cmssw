#! /usr/bin/env python3

import sys
import json

usage = """Usage: mergeResourceJson.py FILE [FILE ...]

Merge the content of multiple "resources.json" files produced by the FastTimerService,
and print the result to standard output.

Example:
    mergeResourceJson.py step*/pid*/resources.json > resources.json
"""
 
def merge_into(metrics, data, dest):
  dest["events"] += data["events"]
  for metric in metrics:
    dest[metric] += data[metric]

def convert_old_resources(resources, filename):
  mapping = {
    "time_real": {"name": "time_real", "description": "real time", "unit": "ms", "title": "Time"},
    "time_thread": {"name": "time_thread", "description": "cpu time", "unit": "ms", "title": "Time"},
    "mem_alloc": {"name": "mem_alloc", "description": "allocated memory", "unit": "kB", "title": "Memory"},
    "mem_free": {"name": "mem_free", "description": "deallocated memory", "unit": "kB", "title": "Memory"}
  }
  new_resources = []
  for resource in resources:
    # check if the keys "name", "description", "unit" and "title" are present
    if all(key in resource for key in ["name", "description", "unit", "title"]):
      new_resources.append(resource)
    elif any(key in resource for key in ["name", "description", "unit", "title"]):
      print("Error: incomplete resource description in file " + filename)
      sys.exit(1)
    else:
      for key, _ in resource.items():
        if key in mapping:
          new_resources.append(mapping[key])
        else:
          new_resources.append(resource)
  return new_resources

if len(sys.argv) == 1:
  print(usage)
  sys.exit(1)

if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
  print(usage)
  sys.exit(0)

with open(sys.argv[1]) as f:
  output = json.load(f)

output["resources"] = convert_old_resources(output["resources"], sys.argv[1])

metrics = []
for resource in output["resources"]:
  if "name" in resource:
    metrics.append(resource["name"])
  else:
    for key in resource:
      metrics.append(key)

datamap = { module["type"] + '|' + module["label"] : module for module in output["modules"] }

for arg in sys.argv[2:]:
  with open(arg) as f:
    input = json.load(f)

  input["resources"] = convert_old_resources(input["resources"], arg)

  if output["resources"] != input["resources"]:
    print("Error: input files describe different metrics")
    sys.exit(1)

  if output["total"]["label"] != input["total"]["label"]:
    print("Warning: input files describe different process names")
  merge_into(metrics, input["total"], output["total"])

  for module in input["modules"]:
    key = module["type"] + '|' + module["label"]
    if key in datamap:
      merge_into(metrics, module, datamap[key])
    else:
      datamap[key] = module
      output["modules"].append(datamap[key])

json.dump(output, sys.stdout, indent = 2 )
sys.stdout.write('\n')
