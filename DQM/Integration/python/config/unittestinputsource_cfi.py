from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import FWCore.ParameterSet.Config as cms

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import fnmatch
from future.moves import subprocess
from .dqmPythonTypes import *

# part of the runTheMatrix magic
from Configuration.Applications.ConfigBuilder import filesFromDASQuery

# This source will process last eventsPerLumi in each provided lumisection.

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
                 355380,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number. This run number has to be present in the dataset configured with the dataset option.")

options.register('dataset',
                 '/ExpressPhysics/Run2022B-Express-v1/FEVT',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Dataset name like '/ExpressCosmics/Commissioning2021-Express-v1/FEVT'")

options.register('maxLumi',
                 20,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Only lumisections up to maxLumi are processed.")

options.register('minLumi',
                 19,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Only lumisections starting from minLumi are processed.")

options.register('lumiPattern',
                 '*',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Only lumisections with numbers matching lumiPattern are processed.")

options.register('eventsPerLumi',
                 100,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "This number of last events in each lumisection will be processed.")

options.register('BeamSplashRun',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Set client source settings for beam SPLASH run")

# This is used only by the online clients themselves. 
# We need to register it here because otherwise an error occurs saying that there is an unidentified option.
options.register('unitTest',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Required to avoid the error.")

options.register('noDB',
                 True, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Don't upload the BeamSpot conditions to the DB")

options.parseArguments()

print("Querying DAS for files...")
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
eventsToProcess = []

# Query DAS for a ROOT file for every lumisection
for ls in range(options.minLumi, options.maxLumi+1):
  if fnmatch.fnmatch(str(ls), options.lumiPattern):
    read, sec = filesFromDASQuery("file run=%d dataset=%s lumi=%s" % (options.runNumber, options.dataset, ls))
    readFiles.extend(read)
    secFiles.extend(sec)

    # Get last eventsPerLumi of events in this file
    command = "edmFileUtil --events %s | tail -n +9 | head -n -5 | awk '{ print $3 }'" % read[0]
    print(command)
    events = subprocess.check_output(command, shell=True)
    events = events.split(b'\n')
    events = filter(lambda x: x != b"", events)
    events = map(int, events)
    events = sorted(events)
    events = events[-options.eventsPerLumi:]
    eventsToProcess.append("%s:%s:%s-%s:%s:%s" % (options.runNumber, ls, events[0], options.runNumber, ls, events[-1]))

eventRange = cms.untracked.VEventRange(eventsToProcess)

print("Got %d files." % len(readFiles))

source = cms.Source ("PoolSource",
                     fileNames = readFiles,
                     secondaryFileNames = secFiles,
                     eventsToProcess = eventRange,
                     ### As we are testing with FEVT, we don't want any unpacked collection
                     ### This makes the tests slightly more realistic (live production uses streamer files
                     inputCommands = cms.untracked.vstring(
                       'drop *',
                       'keep FEDRawDataCollection_rawDataCollector_*_*',
                       'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
                       'keep edmTriggerResults_TriggerResults_*_*'
                     ),
                     dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
                   )
maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

runType = RunType()
if not options.runkey.strip():
  options.runkey = "pp_run"

runType.setRunType(options.runkey.strip())
