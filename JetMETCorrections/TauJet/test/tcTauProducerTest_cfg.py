import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("TCTauRECO")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'rfio:/castor/cern.ch/user/s/slehti/testData/Ztautau_GEN_SIM_RECO_MC_31X_V2_preproduction_311_v1.root'
#    "rfio:/castor/cern.ch/user/s/slehti/testData/MinimumBias_BeamCommissioning09_SD_AllMinBias_Dec19thSkim_336p3_v1_RAWRECO_10ev.root"
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("JetMETCorrections/TauJet/TCTauProducer_cff")

process.runTCTauProducer = cms.Path(
    process.TCTau
)

process.TESTOUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *"
    ),
    fileName = cms.untracked.string('file:testout.root')
)
process.outpath = cms.EndPath(process.TESTOUT)
