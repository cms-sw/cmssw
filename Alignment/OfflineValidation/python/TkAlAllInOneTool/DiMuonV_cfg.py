from fnmatch import fnmatch
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.Utilities.FileUtils as FileUtils
from FWCore.ParameterSet.VarParsing import VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_DoubleMuon_string

import sys
import json
import os

###################################################################
# Define process
###################################################################
process = cms.Process("DiMuonVertexValidation")

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

isMC = config["validation"].get("ismc", True)

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
                                skipEvents = cms.untracked.uint32(0))
else:
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(filesDefaultMC_DoubleMuon_string),
                                skipEvents = cms.untracked.uint32(0))

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

if isMC:
    pass
else:
    process.source.lumisToProcess = goodLumiSecs

###################################################################
## efault set to 1 for unit tests
###################################################################
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(config["validation"].get("maxevents", 100)))

###################################################################
# Bookeeping
###################################################################
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"),
   fileMode  =  cms.untracked.string('NOMERGE'),
)

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.TrackRefitter=dict()
process.MessageLogger.PrimaryVertexProducer=dict()
process.MessageLogger.DiMuonVertexValidation=dict()
process.MessageLogger.DiLeptonHelpCounts=dict()
process.MessageLogger.PlotsVsKinematics=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100)
                                   ),                                                      
    DiMuonVertexValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    DiLeptonHelpCounts = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

###################################################################
# import of standard configurations
###################################################################
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

###################################################################
# TransientTrack from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTransientTracks
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

####################################################################
# Global tag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
# default to remain in sycn with the default input sample
process.GlobalTag = GlobalTag(process.GlobalTag, config["alignment"].get("globaltag", "125X_mcRun3_2022_realistic_v3"))

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
# refitting the muon tracks
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.refittedMuons = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = config["validation"].get("muonTrackcollection", "ALCARECOTkAlDiMuon"), # ALCARECOTkAlDiMuon
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = config["validation"].get("tthrbuilder", "WithAngleAndTemplate"))

###################################################################
# refitting the vertex tracks
###################################################################
process.refittedVtxTracks = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = config["validation"].get("trackcollection", "generalTracks"), # ALCARECOTkAlDiMuonVertexTracks
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = config["validation"].get("tthrbuilder", "WithAngleAndTemplate"))

####################################################################
# Re-do vertices
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
process.offlinePrimaryVerticesFromRefittedTrks = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel = cms.InputTag("refittedVtxTracks")

####################################################################
# BeamSpot check
####################################################################
from RecoVertex.BeamSpotProducer.beamSpotCompatibilityChecker_cfi import beamSpotCompatibilityChecker
process.BeamSpotChecker = beamSpotCompatibilityChecker.clone(
    bsFromEvent = "offlineBeamSpot::RECO",  # source of the event beamspot (in the ALCARECO files)
    bsFromDB = "offlineBeamSpot",           # source of the DB beamspot (from Global Tag) NOTE: only if dbFromEvent is True!
    warningThr = config["validation"].get("bsIncompatibleWarnThresh", 3), # significance threshold to emit a warning message
    errorThr = config["validation"].get("bsIncompatibleErrThresh", 5),    # significance threshold to abort the job
)

####################################################################
# Sequence
####################################################################
process.seqRefitting = cms.Sequence(process.offlineBeamSpot   +
                                    process.BeamSpotChecker   +
                                    process.refittedMuons     +
                                    process.refittedVtxTracks +
                                    process.offlinePrimaryVerticesFromRefittedTrks)

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("{}/DiMuonVertexValidation.root".format(config.get("output", os.getcwd()))),
                                   closeFileFast = cms.untracked.bool(True))

####################################################################
# Analysis module
####################################################################
from Alignment.OfflineValidation.diMuonVertexValidation_cfi import diMuonVertexValidation
process.DiMuonVertexValidation = diMuonVertexValidation.clone(useReco = config["validation"].get("useReco",True),
                                                              vertices = 'offlinePrimaryVerticesFromRefittedTrks')

## the two sets of parameters below are mutually exclusive,
## depending if RECO or ALCARECO is used
## the useReco flag above must be set accordingly
if (config["validation"].get("useReco",True)):
    print("I AM USING RECO DATA-TIER")
    process.DiMuonVertexValidation.muons  = 'muons'
    process.DiMuonVertexValidation.tracks = 'refittedVtxTracks'
else:
    print("I AM USING ALCARECO DATA-TIER")
    if(hasattr(process.DiMuonVertexValidation,'muons')):
        delattr(process.DiMuonVertexValidation,'muons')
    process.DiMuonVertexValidation.muonTracks = cms.InputTag('refittedMuons')

####################################################################
# Path
####################################################################
process.p = cms.Path(process.seqRefitting                            +
                     process.offlinePrimaryVerticesFromRefittedTrks  +
                     process.DiMuonVertexValidation)

print("# Done")
