# Example config to produce a merged bad components payload, e.g.
# cmsRun makeMergeBadComponentPayload_example_cfg.py globalTag=auto:run3_data_prompt runNumber=319176 dqmFile=/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/ZeroBias/R0003191xx/DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root runStartTime=6574046031825076224

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing as VP

process = cms.Process("BadChannelMerge")

opts = VP("python")
opts.register("globalTag", "DONOTEXIST", VP.multiplicity.singleton, VP.varType.string, "GlobalTag")
opts.register("dqmFile", "", VP.multiplicity.singleton, VP.varType.string, "DQM root file")
opts.register("runNumber", 0, VP.multiplicity.singleton, VP.varType.int, "run number")
opts.register("runStartTime", 0, VP.multiplicity.singleton, VP.varType.int, "run start time")
opts.register("dbfile", "merged.db", VP.multiplicity.singleton, VP.varType.string, "SQLite output file")
opts.register("outputTag", "SiStripBadComponents_merged_v0", VP.multiplicity.singleton, VP.varType.string, "Output tag name")
opts.parseArguments()

notAllSet = False
if opts.globalTag == "DONOTEXIST":
    print("ERROR: Global tag must be set")
    notAllSet = True
if opts.runStartTime == 0:
    print("ERROR: Run start time must be set (use the getRunStartTime.py script to get it)")
    notAllSet = True
if opts.runNumber == 0:
    print("ERROR: Run number must be set")
if opts.dqmFile == "":
    print("WARNING: no DQM file set, bad components from FED errors will not be included")
    notAllSet = True
if notAllSet:
    raise RuntimeError("Not all required arguments have been passed, need globalTag, dqmFile, runNumber and runStartTime")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("cout", "cerr", "MergedBadComponents"),
    cerr=cms.untracked.PSet(
        threshold=cms.untracked.string("ERROR")),
    cout=cms.untracked.PSet(
        threshold=cms.untracked.string("INFO"),
        default=cms.untracked.PSet(limit=cms.untracked.int32(0))
        ),
    MergedBadComponents=cms.untracked.PSet(
        threshold=cms.untracked.string("INFO"),
        default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
        SiStripQualityStatistics=cms.untracked.PSet(limit=cms.untracked.int32(100000)),
        ),
    categories = cms.untracked.vstring(
        "SiStripQualityStatistics",
        "SiStripQuality"
        ),
    debugModules = cms.untracked.vstring(
        "SiStripQualityESProducer",
        "siStripBadStripFromQualityDBWriter"
        ),
)

process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, opts.globalTag, "")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(opts.runNumber),
    numberEventsInRun = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    firstTime = cms.untracked.uint64(opts.runStartTime),
    timeBetweenEvents = cms.untracked.uint64(1)
    )
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(record=cms.string("SiStripDetVOffRcd"), tag=cms.string("")),    # DCS information
    cms.PSet(record=cms.string("SiStripDetCablingRcd"), tag=cms.string("")), # Use Detector cabling information to exclude detectors not connected            
    cms.PSet(record=cms.string("SiStripBadChannelRcd"), tag=cms.string("")), # Online Bad components
    cms.PSet(record=cms.string("RunInfoRcd"), tag=cms.string("")),           # List of FEDs exluded during data taking          
    cms.PSet(record=cms.string("SiStripBadFiberRcd"), tag=cms.string("")),   # Bad Channel list from the selected IOV as done at PCL
    )
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)
process.siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)
process.siStripQualityESProducer.PrintDebugOutput = True

# common config for adding bad components from FED errors
from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
badCompFromFedErrors = siStripQualityStatistics.BadComponentsFromFedErrors.clone(
        Add=cms.bool(True),
        LegacyDQMFile=cms.string(opts.dqmFile),
        FileRunNumber=cms.uint32(opts.runNumber)
        )

# Print list of Bad modules and create Tracker Map indicating Bad modules
process.load("DQM.SiStripCommon.TkHistoMap_cff")  # to produce a tracker map
process.stat = siStripQualityStatistics.clone(
        TkMapFileName=cms.untracked.string("TkMap_Jul04_2018_319176.png"),  #available filetypes: .pdf .png .jpg .svg
        BadComponentsFromFedErrors=badCompFromFedErrors
        )

# Write Information into DB
process.load("CalibTracker.SiStripQuality.siStripBadStripFromQualityDBWriter_cfi")
#process.siStripBadStripFromQualityDBWriter.OpenIovAt = cms.untracked.string("currentTime")
process.siStripBadStripFromQualityDBWriter.OpenIovAt = cms.untracked.string("beginTime")
process.siStripBadStripFromQualityDBWriter.BadComponentsFromFedErrors = badCompFromFedErrors
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName=cms.untracked.string("TBufferBlobStreamingService"),
    DBParameters=cms.PSet(
        authenticationPath=cms.untracked.string("/afs/cern.ch/cms/DB/conddb")
    ),
    timetype=cms.untracked.string("runnumber"),
    connect=cms.string("sqlite_file:"+opts.dbfile),
    toPut=cms.VPSet(cms.PSet(
        record=cms.string("SiStripBadModuleRcd"),
        tag=cms.string(opts.outputTag)
    )))
process.siStripBadStripFromQualityDBWriter.record = process.PoolDBOutputService.toPut[0].record

process.p = cms.Path(process.stat*process.siStripBadStripFromQualityDBWriter)
