import FWCore.ParameterSet.Config as cms

alCaHcalPhiSymStream = cms.EDFilter("HLTHcalPhiSymFilter",
    eCut_HE = cms.double(-10.0),
    eCut_HF = cms.double(-10.0),
    eCut_HB = cms.double(-10.0),
    eCut_HO = cms.double(-10.0),
    phiSymHOHitCollection = cms.string('phiSymHcalRecHitsHO'),
    HFHitCollection = cms.InputTag("hfreco"),
    phiSymHBHEHitCollection = cms.string('phiSymHcalRecHitsHBHE'),
    HOHitCollection = cms.InputTag("horeco"),
    phiSymHFHitCollection = cms.string('phiSymHcalRecHitsHF'),
    HBHEHitCollection = cms.InputTag("hbhereco"),
    saveTags = cms.bool( False )
)


