import FWCore.ParameterSet.Config as cms

hltL3fL1TkTripleMu533L31055DZFiltered0p2 = cms.EDFilter("HLT2MuonMuonDZ",
    MaxDZ = cms.double(0.2),
    MinDR = cms.double(0.001),
    MinN = cms.int32(3),
    MinPixHitsForDZ = cms.int32(1),
    checkSC = cms.bool(False),
    inputTag1 = cms.InputTag("hltL3fL1TkTripleMu533PreFiltered555"),
    inputTag2 = cms.InputTag("hltL3fL1TkTripleMu533PreFiltered555"),
    originTag1 = cms.VInputTag("hltPhase2L3MuonCandidates"),
    originTag2 = cms.VInputTag("hltPhase2L3MuonCandidates"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(83),
    triggerType2 = cms.int32(83)
)
