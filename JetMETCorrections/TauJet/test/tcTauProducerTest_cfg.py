import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("TCTauRECO")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_3_6_0_pre1/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0002/1E8AE923-2922-DF11-B460-0030487CD7B4.root',
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
process.GlobalTag.globaltag = cms.string('START36_V2::All')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

#process.load("JetMETCorrections/TauJet/TCTauProducer_cff")
process.load("RecoTauTag.Configuration.RecoTauTag_cff")

process.runTCTauProducer = cms.Path(
    process.tautagging
)

process.TESTOUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *"
    ),
    fileName = cms.untracked.string('file:testout.root')
)
process.outpath = cms.EndPath(process.TESTOUT)
