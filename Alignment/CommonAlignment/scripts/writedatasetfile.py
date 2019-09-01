#!/usr/bin/env python

import argparse

from Alignment.CommonAlignment.tools.dataset import MultipleDatasets

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="dataset name on DAS (can give multiple)", action="append")
parser.add_argument("--hippy", help="output file name for HipPy")
parser.add_argument("-v", "--validation", help="output file name for validation")
parser.add_argument("-f", "--first-run", default=0, type=int)
parser.add_argument("-l", "--last-run", default=float("inf"), type=int)
parser.add_argument("-r", "--run", action="append", type=int)
parser.add_argument("-m", "--max-events", default=float("inf"), type=int)
parser.add_argument("-j", "--events-per-job", type=int)
parser.add_argument("--clear-cache", action="store_true")

args = parser.parse_args()
dataset = MultipleDatasets(*args.dataset)

if not args.validation and not args.hippy:
  raise RuntimeError("have to provide --validation or --hippy")
if args.validation and not args.hippy and args.events_per_job:
  raise RuntimeError("--events-per-job is only used for HipPy")

if args.hippy:
  dataset.writefilelist_hippy(firstrun=args.first_run, lastrun=args.last_run, runs=args.run, maxevents=args.max_events, eventsperjob=args.events_per_job, outputfile=args.hippy, usecache=not args.clear_cache)
if args.validation:
  dataset.writefilelist_validation(firstrun=args.first_run, lastrun=args.last_run, runs=args.run, maxevents=args.max_events, outputfile=args.validation, usecache=not args.clear_cache)
