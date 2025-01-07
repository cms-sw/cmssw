import json
import os
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_NoPU
from FWCore.ParameterSet.VarParsing import VarParsing

###################################################################
# Define process 
###################################################################
process = cms.Process("GenericTrackAndVertexValidation")

###################################################################
# Argument parsing
###################################################################
options = VarParsing()
options.register("config", "", VarParsing.multiplicity.singleton, VarParsing.varType.string , "AllInOne config")
options.parseArguments()

###################################################################
# Read in AllInOne config in JSON format
###################################################################
if options.config == "":
    config = {"validation": {},
              "alignment": {}}
else:
    with open(options.config, "r") as configFile:
        config = json.load(configFile)

###################################################################
# Read filenames from given TXT file and define input source
###################################################################
readFiles = []

if "dataset" in config["validation"]:
    with open(config["validation"]["dataset"], "r") as datafiles:
        for fileName in datafiles.readlines():
            readFiles.append(fileName.replace("\n", ""))

    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(readFiles),
                                skipEvents = cms.untracked.uint32(0)
                                )
else:
    print(">>>>>>>>>> GenericV_cfg.py: msg%-i: config not specified! Loading default MC simulation -> filesDefaultMC_NoPU!")
    process.source = cms.Source("PoolSource",
                                fileNames = filesDefaultMC_NoPU,
                                skipEvents = cms.untracked.uint32(0)
                                ) 

###################################################################
# Get good lumi section and load data or handle MC
###################################################################
if "goodlumi" in config["validation"]:
     if os.path.isfile(config["validation"]["goodlumi"]):
          goodLumiSecs = cms.untracked.VLuminosityBlockRange(LumiList.LumiList(filename = config["validation"]["goodlumi"]).getCMSSWString().split(','))
     else:
          print("Does not exist: {}. Continue without good lumi section file.")
          goodLumiSecs = cms.untracked.VLuminosityBlockRange()

else:
     goodLumiSecs = cms.untracked.VLuminosityBlockRange()

###################################################################
# Runs and events
###################################################################
runboundary = config["validation"].get("runboundary", 1)
isMultipleRuns=False
if(isinstance(runboundary, (list, tuple))):
     isMultipleRuns=True
     print("Multiple Runs are selected")
if(isMultipleRuns):
     process.source.firstRun = cms.untracked.uint32(runboundary[0])
else:
     process.source.firstRun = cms.untracked.uint32(runboundary)

###################################################################
# Default set to 1 for unit tests
###################################################################
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(config["validation"].get("maxevents", 1)))

###################################################################
# Bookeeping
###################################################################
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
    fileMode  =  cms.untracked.string('NOMERGE'), # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

###################################################################
# Basic modules
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

####################################################################
# Load and Configure Track refitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = config["validation"].get("trackcollection", "generalTracks")
process.TrackRefitter.TTRHBuilder = config["validation"].get("tthrbuilder", "WithAngleAndTemplate")
process.TrackRefitter.NavigationSchool = ""

####################################################################
# Global tag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, config["alignment"].get("globaltag", "auto:phase1_2017_realistic"))

####################################################################
# Load conditions if wished
####################################################################
if "conditions" in config["alignment"]:
    from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource

    for condition in config["alignment"]["conditions"]:
        setattr(process, "conditionsIn{}".format(condition), poolDBESSource.clone(
             connect = cms.string(str(config["alignment"]["conditions"][condition]["connect"])),
             toGet = cms.VPSet(
                        cms.PSet(
                                 record = cms.string(str(condition)),
                                 tag = cms.string(str(config["alignment"]["conditions"][condition]["tag"]))
                        )
                     )
            )
        )

        setattr(process, "prefer_conditionsIn{}".format(condition), cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition)))

###################################################################
# TFileService
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("{}/GenericValidation.root".format(config.get("output", os.getcwd()))),
                                   closeFileFast = cms.untracked.bool(True))

print("Saving the output at %s" % process.TFileService.fileName.value())

###################################################################
# Primary Vertex refit
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices as _offlinePVs
process.offlinePrimaryVerticesFromRefittedTrks = _offlinePVs.clone(
    TrackLabel = "TrackRefitter",
    TkFilterParameters = dict(
        maxNormalizedChi2 = 20,
        minSiliconLayersWithHits = 5,
        maxD0Significance = 5.0,
        maxD0Error = 1.0,
        maxDzError = 1.0,
        minPixelLayersWithHits = 2
    )
)

###################################################################
# The analysis modules
###################################################################
process.trackanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                       TkTag  = cms.InputTag("TrackRefitter"),
                                       isCosmics = cms.bool(config["validation"].get("isCosmics", False))
                                       )

process.vertexanalysis = cms.EDAnalyzer('GeneralPurposeVertexAnalyzer',
                                        ndof = cms.int32(4),
                                        vertexLabel = cms.InputTag('offlinePrimaryVerticesFromRefittedTrks'),
                                        beamSpotLabel = cms.InputTag('offlineBeamSpot'),
                                        Xpos = cms.double(0.1),
                                        Ypos = cms.double(0),
                                        TkSizeBin = cms.int32(100),
                                        TkSizeMin = cms.double(499.5),
                                        TkSizeMax = cms.double(-0.5),
                                        DxyBin = cms.int32(100),
                                        DxyMin = cms.double(5000),
                                        DxyMax = cms.double(-5000),
                                        DzBin = cms.int32(100),
                                        DzMin = cms.double(-2000),
                                        DzMax = cms.double(2000),
                                        PhiBin = cms.int32(32),
                                        PhiBin2D = cms.int32(12),
                                        PhiMin = cms.double(-3.1415926535897931),
                                        PhiMax = cms.double(3.1415926535897931),
                                        EtaBin = cms.int32(26),
                                        EtaBin2D = cms.int32(8),
                                        EtaMin = cms.double(-2.7),
                                        EtaMax = cms.double(2.7))

process.theValidSequence = cms.Sequence(process.offlineBeamSpot +
                                        process.TrackRefitter   +
                                        process.offlinePrimaryVerticesFromRefittedTrks +
                                        process.trackanalysis   +
                                        process.vertexanalysis)

###################################################################
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.theHLTFilter = triggerResultsFilter.clone(
    triggerConditions = cms.vstring(config["validation"].get("triggerBits", "*")),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
)

HLTSel = config["validation"].get("HLTselection", False)

if (HLTSel):
    process.p = cms.Path(process.theHLTFilter + process.theValidSequence)
else:
    process.p = cms.Path(process.theValidSequence)

print("Done")
