import FWCore.ParameterSet.Config as cms

HLTEff = cms.EDAnalyzer("HLTEffCalculator",
    TriggerResCollection = cms.InputTag("TriggerResults","","HLT"), 
    hltPaths = cms.vstring('HLT_IsoMu24_v15','HLT_IsoMu24_eta2p1_v13','HLT_IsoMu30_v9'),
    OutputFileName = cms.string("triggerEfficiency.root"),
    verbosity = cms.untracked.int32(0)
 )
