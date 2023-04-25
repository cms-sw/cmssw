import json
import os
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_NoPU
from FWCore.ParameterSet.VarParsing import VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.utils import _byteify

###################################################################
# Define process 
###################################################################
process = cms.Process("PrimaryVertexResolution")

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
    print(">>>>>>>>>> SplitV_cfg.py: msg%-i: config not specified! Loading default MC simulation -> filesDefaultMC_NoPU!")
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
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.theHLTFilter = triggerResultsFilter.clone(
    triggerConditions = cms.vstring(config["validation"].get("triggerBits", "*")),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
)

###################################################################
# PV refit
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices 
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("TrackRefitter") 
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0
# as it was prior to https://github.com/cms-sw/cmssw/commit/c8462ae4313b6be3bbce36e45373aa6e87253c59
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Error                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxDzError                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207

###################################################################
# The PV resolution module
###################################################################
from Alignment.OfflineValidation.splitVertexResolution_cfi import splitVertexResolution as _splitVertexResolution
process.PrimaryVertexResolution = _splitVertexResolution.clone(
    compressionSettings = compressionSettings,
    storeNtuple         = False,
    vtxCollection       = "offlinePrimaryVerticesFromRefittedTrks",
    trackCollection     = "TrackRefitter",
    minVertexNdf        = 10.,
    minVertexMeanWeight = 0.5,
    runControl          = config["validation"].get("runControl", False),
    runControlNumber    = [runboundary]
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("{}/SplitV.root".format(config.get("output", os.getcwd()))),
                                   closeFileFast = cms.untracked.bool(True)
                                   )
print("Saving the output at %s" % process.TFileService.fileName.value())

process.theValidSequence = cms.Sequence(process.offlineBeamSpot                        +
                                        process.TrackRefitter                          +
                                        process.offlinePrimaryVerticesFromRefittedTrks +
                                        process.PrimaryVertexResolution)

HLTSel = config["validation"].get("HLTselection", False)

if (HLTSel):
    process.p = cms.Path(process.theHLTFilter + process.theValidSequence)
else:
    process.p = cms.Path(process.theValidSequence)

print("Done")
