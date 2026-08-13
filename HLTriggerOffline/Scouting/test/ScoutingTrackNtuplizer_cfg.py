import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')

options.register('globalTag',
                 '160X_dataRun3_HLT_v1',  # adjust to your run era
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 'Global tag')
options.parseArguments()

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
process = cms.Process('DZDEBUG',Run3_2025)

# ---- number of events -------------------------------------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# ---- message logger (keep it quiet) -----------------------------------------
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# ---- global tag -------------------------------------------------------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

# ---- geometry & magnetic field (needed to build reco::Track properly) -------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# ---- input ------------------------------------------------------------------
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(options.inputFiles),
)

# ---- TFileService (output TTree goes here) ----------------------------------
process.TFileService = cms.Service('TFileService',
    fileName = cms.string(options.outputFile),
    closeFileFast = cms.untracked.bool(True)
)

# ---- the debugger module ----------------------------------------------------
process.scoutingTrackNtuplizer = cms.EDAnalyzer('ScoutingTrackNtuplizer',
    tracks        = cms.InputTag('hltScoutingTrackPacker'),
    vertices      = cms.InputTag('hltScoutingPrimaryVertexPacker', 'primaryVtx'),
    beamSpotLabel = cms.InputTag('hltOnlineBeamSpot'),
)

# ---- the beamspot module ----------------------------------------------------
from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer as _onlineBeamSpotProducer
process.hltOnlineBeamSpot = _onlineBeamSpotProducer.clone()

# ---- path -------------------------------------------------------------------
process.p = cms.Path(process.hltOnlineBeamSpot+process.scoutingTrackNtuplizer)
process.schedule = cms.Schedule(process.p)
