import FWCore.ParameterSet.Config as cms

hltBTagPFPuppiDeepCSV0p865DoubleEta2p4 = cms.EDFilter("HLTPFJetTag",
    JetTags = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPFPuppiModEta2p4","probb"),
    Jets = cms.InputTag("hltPFPuppiJetForBtagEta2p4"),
    MatchJetsByDeltaR = cms.bool(True),
    MaxJetDeltaR = cms.double(0.1),
    MaxTag = cms.double(999999.0),
    MinJets = cms.int32(2),
    MinTag = cms.double(0.865),
    TriggerType = cms.int32(86),
    saveTags = cms.bool(True)
)
