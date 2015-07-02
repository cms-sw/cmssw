import FWCore.ParameterSet.Config as cms

cmgMuEle = cms.EDProducer(
    "MuElePOProducer",
    leg1Collection=cms.InputTag('muonPreSelectionMuEle'),
    leg2Collection=cms.InputTag('electronPreSelectionMuEle'),
    metCollection=cms.InputTag('mvaMETMuEle'),
    )
