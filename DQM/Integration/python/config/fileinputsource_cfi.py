import FWCore.ParameterSet.Config as cms

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
import fnmatch
from dqmPythonTypes import *

from dataset_cfi import *

options = VarParsing.VarParsing("analysis")

options.register(
    "runkey",
    "pp_run",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Run Keys of CMS"
)

options.register('runNumber',
                 286520,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number. This run number has to be present in the dataset configured in dataset_cfi.py.")

options.register('maxLumi',
                 2000,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Only lumisections up to maxLumi are processed.")

options.register('lumiPattern',
                 '*0',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Only lumisections with numbers matching lumiPattern are processed.")

options.parseArguments()

runstr = str(options.runNumber)
runpattern = "*" + runstr[0:3] + "/" + runstr[3:] + "*"
readFiles = cms.untracked.vstring([f for f in readFiles if fnmatch.fnmatch(f, runpattern)])
lumirange =  cms.untracked.VLuminosityBlockRange(
  [ str(options.runNumber) + ":" + str(ls) 
      for ls in range(1, options.maxLumi+1) 
      if fnmatch.fnmatch(str(ls), options.lumiPattern)
  ]
)

print "Selected %d files and %d LS" % (len(readFiles), len(lumirange)) 

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
