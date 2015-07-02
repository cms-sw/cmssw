import FWCore.ParameterSet.Config as cms

cmgDiTau = cms.EDProducer(
    "DiTauPOProducer",
    leg1Collection = cms.InputTag("tauPreSelectionDiTau"),
    leg2Collection = cms.InputTag("tauPreSelectionDiTau"),
    metCollection = cms.InputTag('mvaMETDiTau') 
    )
