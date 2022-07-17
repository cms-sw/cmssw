import FWCore.ParameterSet.Config as cms

hltRechitInRegionsHGCAL = cms.EDProducer("HLTHGCalRecHitsInRegionsProducer",
    etaPhiRegions = cms.VPSet(cms.PSet(
        inputColl = cms.InputTag("hltL1TEGammaHGCFilteredCollectionProducer"),
        maxDEta = cms.double(0.0),
        maxDPhi = cms.double(0.0),
        maxDeltaR = cms.double(0.35),
        maxEt = cms.double(999999.0),
        minEt = cms.double(5.0),
        type = cms.string('L1EGamma')
    )),
    inputCollTags = cms.VInputTag("HGCalRecHitL1Seeded:HGCEERecHits", "HGCalRecHitL1Seeded:HGCHEBRecHits", "HGCalRecHitL1Seeded:HGCHEFRecHits"),
    outputProductNames = cms.vstring(
        'HGCEERecHits',
        'HGCHEBRecHits',
        'HGCHEFRecHits'
    )
)
