import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        #"/store/data/Run2016B/SingleMuon/RAW/v2/000/274/161/00000/02B65CDB-1B25-E611-AABD-02163E0135F0.root",
        #"/store/data/Run2016B/SingleMuon/RAW/v2/000/274/161/00000/00098E23-3825-E611-A603-02163E0134BD.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/001AF1D0-6BE9-E511-9A8D-02163E0143FE.root", 
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/005BCAC3-72E9-E511-B002-02163E014310.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/00E3C803-BBE9-E511-A99C-02163E01465A.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/00EFE5C2-6EE9-E511-ABB3-02163E013417.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/04159D15-95E9-E511-B73F-02163E014176.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/04E4A62F-93E9-E511-83A3-02163E0134CD.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/04F7F2D7-72E9-E511-9534-02163E014310.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/06077D63-78E9-E511-9D8A-02163E014310.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/061EFCD5-97E9-E511-8764-02163E0143FE.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/063C23DD-82E9-E511-9020-02163E011FCE.root",        
        "file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root",
        #"/store/data/Commissioning2014/Cosmics/RAW//v3/000/224/380/00000/E05943D1-1227-E411-BB8E-02163E00F0C4.root",
        #"/store/data/Commissioning2014/Cosmics/RAW/v3/000/224/380/00000/68FDADE5-1227-E411-8AA6-02163E00A10C.root",
    )
)

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
import fnmatch
from dqmPythonTypes import *

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

options.parseArguments()

try:
  # fixed dataset, DAS 'py' snippet
  from dataset_cfi import readFiles, secFiles
  print "Using filenames from dataset_cfi.py."
except:
  if options.dataset == 'auto':
    print "Querying DAS for a dataset..."
    import subprocess
    out = subprocess.check_output("das_client --query 'dataset run=%d dataset=/*Express*/*/*FEVT*'" % options.runNumber, shell=True)
    dataset = out.splitlines()[-1]
    print "Using dataset=%s." % dataset
  else:
    dataset = options.dataset

  print "Querying DAS for files..."
  readFiles = cms.untracked.vstring()
  secFiles = cms.untracked.vstring()
  # this outputs all results, which can be a lot...
  read, sec = filesFromDASQuery("file run=%d dataset=%s" % (options.runNumber, dataset), option=" --limit 10000 ")
  readFiles.extend(read)
  secFiles.extend(sec)

print "Got %d files." % len(readFiles)

runstr = str(options.runNumber)
runpattern = "*" + runstr[0:3] + "/" + runstr[3:] + "*"
readFiles = cms.untracked.vstring([f for f in readFiles if fnmatch.fnmatch(f, runpattern)])
lumirange =  cms.untracked.VLuminosityBlockRange(
  [ str(options.runNumber) + ":" + str(ls) 
      for ls in range(options.minLumi, options.maxLumi+1)
      if fnmatch.fnmatch(str(ls), options.lumiPattern)
  ]
)

print "Selected %d files and %d LS." % (len(readFiles), len(lumirange))

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
