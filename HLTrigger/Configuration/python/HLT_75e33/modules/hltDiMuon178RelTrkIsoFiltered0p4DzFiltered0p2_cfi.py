import FWCore.ParameterSet.Config as cms

hltDiMuon178RelTrkIsoFiltered0p4DzFiltered0p2 = cms.EDFilter("HLT2MuonMuonDZ",
    MaxDZ = cms.double(0.2),
    MinDR = cms.double(0.001),
    MinN = cms.int32(1),
    MinPixHitsForDZ = cms.int32(0),
    checkSC = cms.bool(False),
    inputTag1 = cms.InputTag("hltDiMuon178RelTrkIsoFiltered0p4"),
    inputTag2 = cms.InputTag("hltDiMuon178RelTrkIsoFiltered0p4"),
    originTag1 = cms.VInputTag("hltPhase2L3MuonCandidates"),
    originTag2 = cms.VInputTag("hltPhase2L3MuonCandidates"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(83),
    triggerType2 = cms.int32(83)
)
