import FWCore.ParameterSet.Config as cms

hltHgcalDigisMuonSeeded = cms.EDProducer("HLTHGCalDigisInRegionsProducer",
    etaPhiRegions = cms.VPSet(cms.PSet(
        inputColl = cms.InputTag("hltPhase2L3MuonCandidates"),
        maxDEta = cms.double(0.0),
        maxDPhi = cms.double(0.0),
        maxDeltaR = cms.double(0.5),
        maxEt = cms.double(999999.0),
        minEt = cms.double(0.0),
        type = cms.string('RecoChargedCandidate')
    )),
    inputCollTags = cms.VInputTag("hltHgcalDigis:EE", "hltHgcalDigis:HEback", "hltHgcalDigis:HEfront"),
    outputProductNames = cms.vstring(
        'EE',
        'HEback',
        'HEfront'
    )
)
