#!/usr/bin/env python

import os
import collections
import logging
import resource
import time
import argparse
import subprocess
import signal
import json
import inspect
import shutil
import glob

import sys

LOG_FORMAT = '%(asctime)s: %(name)-20s - %(levelname)-8s - %(message)s'
logging.basicConfig(format=LOG_FORMAT)
log = logging.getLogger("mbProfile")
log.setLevel(logging.INFO)


def run_matrix(pargs):
    pargs.extend(["--command", '--customise DQMServices/Components/customize_DQMStreamStats.dumpEndOfRun'])
    proc = subprocess.Popen(pargs)
    proc.wait()


def transfer(args):
    if args.source:
        inputPath = args.source
    else:
        # Find source repository by using workflow matching
        for name in glob.glob(args.pargs[2] + "*"):
            if os.path.exists(name):
                inputPath = name
                break

    if os.path.exists(inputPath):
        transfer_jsons(inputPath, args.folder)
    else:
        log.error("Could not find source repository: %s ", inputPath)


def transfer_jsons(inputPath, outputPath):
    os.chdir(inputPath)
    for file in glob.glob("*.json"):
        source_fn = os.path.join(os.getcwd(), file)
        target_fn = os.path.join('../' + outputPath, file)
        log.info("Copying %s to %s", source_fn, target_fn)
        shutil.copyfile(source_fn, target_fn)

    # Create json file
    profile = os.path.basename(glob.glob("*.json")[0])
    os.chdir('..')
    target_fn = os.path.join(outputPath, "mbGraph.json")
    log.info("Creating %s", target_fn)
    with open(target_fn, "w") as fp:
        dct = {
            "file": profile,
            "env": {
                "CMSSW_GIT_HASH": os.getenv("CMSSW_GIT_HASH"),
                "CMSSW_RELEASE_BASE": os.getenv("CMSSW_RELEASE_BASE"),
                "SCRAM_ARCH": os.getenv("SCRAM_ARCH"),
            },
        }

        json.dump(dct, fp, indent=2)


def find_and_write_html(outputPath):
    # Create the dir if necessary
    if outputPath and not os.path.exists(outputPath):
        os.makedirs(outputPath)

    html_paths = [
        os.path.join(os.getenv("CMSSW_BASE"), "src/DQMServices/Components/data/html/page"),
        os.path.join(os.getenv("CMSSW_RELEASE_BASE"), "src/DQMServices/Components/data/html/page"),
    ]

    def find_file(f):
        fails = []
        for p in html_paths:
            x = os.path.join(p, f)
            if os.path.exists(x):
                return x
            else:
                fails.append(x)

        log.warning("Could not find html file: %s (%s)", f, fails)

    for f in ['mbGraph.js', 'mbGraph.html', 'memoryGraph.js']:
        source_fn = find_file(f)
        target_fn = os.path.join(outputPath, f)
        if source_fn:
            log.info("Copying %s to %s", source_fn, target_fn)
            shutil.copyfile(source_fn, target_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile child processes and produce data for rss and such graphs.")
    parser.add_argument("-f", "--folder", type=str, default="memory", help="Folder name to write.", metavar="memory")
    parser.add_argument('-q', action='store_true', help="Reduce logging.")
    parser.add_argument("-src", "--source", type=str, default=None, help="Source folder path.")
    parser.add_argument('pargs', nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()

    if not args.pargs and not args.source:
        parser.print_help()
        sys.exit(-1)
    elif args.pargs and args.pargs[0] == "--":
        # compat with 2.6
        args.pargs = args.pargs[1:]

    if args.q:
        log.setLevel(logging.WARNING)

    # Copy html and js files
    find_and_write_html(args.folder)

    # If not source folder then run the pargs command with customise
    if not args.source:
        run_matrix(args.pargs)

    # Transfer json files to folder repository
    transfer(args)
