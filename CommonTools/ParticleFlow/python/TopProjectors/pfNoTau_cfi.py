import FWCore.ParameterSet.Config as cms

pfNoTau = cms.EDProducer(
    "TPPFTausOnPFJetsDeltaR",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    deltaR = cms.double( 0.5 ),
    name = cms.untracked.string("noTau"),
    topCollection = cms.InputTag("pfTausPtrs"),
    bottomCollection = cms.InputTag("pfJetsPtrs"),
)
