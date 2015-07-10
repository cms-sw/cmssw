import FWCore.ParameterSet.Config as cms

cmgTauEle = cms.EDProducer(
    "TauElePOProducer",
    leg1Collection=cms.InputTag('tauPreSelectionTauEle'),
    leg2Collection=cms.InputTag('electronPreSelectionTauEle'),
    metCollection=cms.InputTag('mvaMETTauEle'),
    )
