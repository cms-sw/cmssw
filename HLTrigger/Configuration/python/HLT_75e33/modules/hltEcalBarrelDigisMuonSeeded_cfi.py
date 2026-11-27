import FWCore.ParameterSet.Config as cms

hltEcalBarrelDigisMuonSeeded = cms.EDProducer("HLTEcalEBDigisInRegionsProducer",
    etaPhiRegions = cms.VPSet(cms.PSet(
        inputColl = cms.InputTag("hltPhase2L3MuonCandidates"),
        maxDEta = cms.double(0.0),
        maxDPhi = cms.double(0.0),
        maxDeltaR = cms.double(0.45),
        maxEt = cms.double(999999.0),
        minEt = cms.double(0.0),
        type = cms.string('RecoChargedCandidate')
    )),
    inputCollTags = cms.VInputTag("hltEcalDigis:ebDigis"),
    outputProductNames = cms.vstring('ebDigis')
)
