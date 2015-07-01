import FWCore.ParameterSet.Config as cms

cmgDiMu = cms.EDProducer(
    "DiMuPOProducer",
    leg1Collection=cms.InputTag('muonPreSelectionDiMu'),
    leg2Collection=cms.InputTag('muonPreSelectionDiMu'),
    metCollection=cms.InputTag('mvaMETDiMu'),
    )
