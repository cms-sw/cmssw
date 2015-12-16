#!/usr/bin/env python

import os
import sys

if not os.environ.has_key("CMSSW_BASE"):
    print "You need to source the CMSSW environment first."
    sys.exit(1)

required_version = (2,7)
if sys.version_info < required_version:
    print "Your Python interpreter is too old. Need version 2.7 or higher."
    sys.exit(1)

import argparse

import HLTrigger.Tools.rrapi as rrapi
from FWCore.PythonUtilities.LumiList import LumiList


def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Create JSON selection for a given magnetic field.")
    parser.add_argument("-i", "--input", dest="input", metavar="JSON",
                        type=str, help="input JSON file")
    parser.add_argument("-o", "--output", dest="output", metavar="JSON",
                        type=str, help="output JSON file")
    parser.add_argument("--min", dest="min", metavar="RUN", type=int,
                        help="first run to be considered in the selection")
    parser.add_argument("--max", dest="max", metavar="RUN", type=int,
                        help="last run to be considered in the selection")
    parser.add_argument("--epsilon", dest="epsilon", metavar="TESLA",
                        default=0.1, type=float,
                        help="precision of the filter (default: %(default)s T)")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="show more verbose output")
    required = parser.add_argument_group("required arguments")
    required.add_argument("--b-field", dest="bfield", metavar="TESLA",
                          required=True, type=float,
                          help="magnetic field to filter")
    args = parser.parse_args(argv)


    try:
        if args.input == None and (args.min == None or args.max == None):
            msg = ("If no input JSON file ('--input') is provided, you have to "
                   "explicitly provide the first ('--min') and last ('--max') "
                   "run.")
            raise RuntimeError(msg)

        if args.min != None and args.max != None and args.min > args.max:
            msg = "First run ({min:d}) is after last run ({max:d})."
            msg = msg.format(**args.__dict__)
            raise RuntimeError(msg)

        if args.max != None and args.max <= 0:
            msg = "Last run must be greater than zero: max = {0:d} <= 0."
            msg = msg.format(args.max)
            raise RuntimeError(msg)
    except RuntimeError, e:
        if args.debug: raise
        print ">>>", os.path.splitext(os.path.basename(__file__))[0]+":", str(e)
        sys.exit(1)


    lumi_list = None if not args.input else LumiList(filename = args.input)
    input_runs = None if not lumi_list else [int(r) for r in lumi_list.getRuns()]

    # Run registry API: https://twiki.cern.ch/twiki/bin/viewauth/CMS/DqmRrApi
    URL = "http://runregistry.web.cern.ch/runregistry/"
    api = rrapi.RRApi(URL, debug = args.debug)

    if api.app != "user": return

    column_list = ("number",)
    min_run = args.min if args.min != None else input_runs[0]
    max_run = args.max if args.max != None else input_runs[-1]
    bfield_min = args.bfield - args.epsilon
    bfield_max = args.bfield + args.epsilon
    constraints = {
        "datasetExists": "= true",
        "number": ">= {0:d} and <= {1:d}".format(min_run, max_run),
        "bfield": "> {0:f} and < {1:f}".format(bfield_min, bfield_max)
        }

    run_list = [item["number"] for item in
                api.data(workspace = "GLOBAL", table = "runsummary",
                         template = "json", columns = column_list,
                         filter = constraints)]

    if lumi_list != None:
        runs_to_remove = []
        for run in input_runs:
            if run not in run_list: runs_to_remove.append(run)
        lumi_list.removeRuns(runs_to_remove)
    else:
        lumi_list = LumiList(runs = run_list)

    if args.output != None:
        lumi_list.writeJSON(args.output)
        with open(args.output+".args", "w") as f:
            f.write(" ".join(argv)+"\n")
    else:
        print lumi_list



if __name__ == "__main__":
    main()
