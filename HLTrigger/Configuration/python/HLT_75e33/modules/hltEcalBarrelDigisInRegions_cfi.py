import FWCore.ParameterSet.Config as cms

hltEcalBarrelDigisInRegions = cms.EDProducer("HLTEcalEBDigisInRegionsProducer",
    etaPhiRegions = cms.VPSet(cms.PSet(
        inputColl = cms.InputTag("hltL1TEGammaFilteredCollectionProducer"),
        maxDEta = cms.double(0.0),
        maxDPhi = cms.double(0.0),
        maxDeltaR = cms.double(0.35),
        maxEt = cms.double(999999.0),
        minEt = cms.double(5.0),
        type = cms.string('L1EGamma')
    )),
    inputCollTags = cms.VInputTag("hltEcalDigis:ebDigis"),
    outputProductNames = cms.vstring('ebDigis')
)
