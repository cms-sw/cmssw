from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import FWCore.ParameterSet.Config as cms

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
import fnmatch
from .dqmPythonTypes import *

# part of the runTheMatrix magic
from Configuration.Applications.ConfigBuilder import filesFromDASQuery

options = VarParsing.VarParsing("analysis")

options.register(
    "runkey",
    "pp_run",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Run Keys of CMS"
)

# Parameter for frontierKey
options.register('runUniqueKey',
    'InValid',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Unique run key from RCMS for Frontier")

options.register('runNumber',
                 286520,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number. This run number has to be present in the dataset configured with the dataset option.")

options.register('maxLumi',
                 2000,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Only lumisections up to maxLumi are processed.")

options.register('minLumi',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Only lumisections starting from minLumi are processed.")

options.register('lumiPattern',
                 '*0',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Only lumisections with numbers matching lumiPattern are processed.")

options.register('dataset',
                 'auto',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Dataset name like '/ExpressPhysicsPA/PARun2016D-Express-v1/FEVT', or 'auto' to guess it with a DAS query. A dataset_cfi.py that defines 'readFiles' and 'secFiles' (like a DAS Python snippet) will override this, to avoid DAS queries.")

options.register('noDB',
                 True, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Don't upload the BeamSpot conditions to the DB")

options.parseArguments()

try:
  # fixed dataset, DAS 'py' snippet
  from dataset_cfi import readFiles, secFiles
  print("Using filenames from dataset_cfi.py.")
except:
  if options.dataset == 'auto':
    print("Querying DAS for a dataset...")
    import subprocess
    out = subprocess.check_output("dasgoclient --query 'dataset run=%d dataset=/*Express*/*/*FEVT*'" % options.runNumber, shell=True)
    dataset = out.splitlines()[-1]
    print("Using dataset=%s." % dataset)
  else:
    dataset = options.dataset

  print("Querying DAS for files...")
  readFiles = cms.untracked.vstring()
  secFiles = cms.untracked.vstring()
  # this outputs all results, which can be a lot...
  read, sec = filesFromDASQuery("file run=%d dataset=%s" % (options.runNumber, dataset), option=" --limit 10000 ")
  readFiles.extend(read)
  secFiles.extend(sec)

print("Got %d files." % len(readFiles))

runstr = str(options.runNumber)
runpattern = "*" + runstr[0:3] + "/" + runstr[3:] + "*"
readFiles = cms.untracked.vstring([f for f in readFiles if fnmatch.fnmatch(f, runpattern)])
secFiles = cms.untracked.vstring([f for f in secFiles if fnmatch.fnmatch(f, runpattern)])
lumirange =  cms.untracked.VLuminosityBlockRange(
  [ str(options.runNumber) + ":" + str(ls) 
      for ls in range(options.minLumi, options.maxLumi+1)
      if fnmatch.fnmatch(str(ls), options.lumiPattern)
  ]
)

print("Selected %d files and %d LS." % (len(readFiles), len(lumirange)))

source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles, lumisToProcess = lumirange)
maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType()
if not options.runkey.strip():
    options.runkey = "pp_run"

runType.setRunType(options.runkey.strip())
