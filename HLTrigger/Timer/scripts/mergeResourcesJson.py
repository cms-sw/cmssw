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


if len(sys.argv) == 1:
  print(usage)
  sys.exit(1)

if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
  print(usage)
  sys.exit(0)

with open(sys.argv[1]) as f:
  output = json.load(f)

metrics = [ label for resource in output["resources"] for label in resource ]

datamap = { module["type"] + '|' + module["label"] : module for module in output["modules"] }

for arg in sys.argv[2:]:
  with open(arg) as f:
    input = json.load(f)

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
