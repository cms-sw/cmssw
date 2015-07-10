import FWCore.ParameterSet.Config as cms

cmgTauMu = cms.EDProducer(
    "TauMuPOProducer",
    leg1Collection=cms.InputTag('tauPreSelectionTauMu'),
    leg2Collection=cms.InputTag('muonPreSelectionTauMu'),
    metCollection=cms.InputTag('mvaMETTauMu'),
    )
