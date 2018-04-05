#!/usr/bin/env python
from Alignment.OfflineValidation.TkAlAllInOneTool.dataset import Dataset
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("outputfilename", help="Goes into $CMSSW_BASE/src/Alignment/HIPAlignmentAlgorithm/data unless an absolute path starting with / is provided.  example: ALCARECOTkAlMinBias.dat_example")
parser.add_argument("datasetname", help="example: /ZeroBias/Run2016G-TkAlMinBias-PromptReco-v1/ALCARECO")
parser.add_argument("filesperjob", type=int, help="max number of files in each job")
parser.add_argument("firstrun", type=int, nargs="?", help="first run to use")
parser.add_argument("lastrun", type=int, nargs="?", help="last run to use")
args = parser.parse_args()

dataset = Dataset(args.datasetname, tryPredefinedFirst=False)
outputfilename = os.path.join(os.environ["CMSSW_BASE"], "src", "Alignment", "HIPAlignmentAlgorithm", "data", args.outputfilename)
dataset.createdatasetfile_hippy(outputfilename, args.filesperjob, args.firstrun, args.lastrun)
