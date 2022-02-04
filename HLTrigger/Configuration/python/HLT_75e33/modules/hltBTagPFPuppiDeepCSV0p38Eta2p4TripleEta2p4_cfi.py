import FWCore.ParameterSet.Config as cms

hltBTagPFPuppiDeepCSV0p38Eta2p4TripleEta2p4 = cms.EDFilter("HLTPFJetTag",
    JetTags = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPFPuppiModEta2p4","probb"),
    Jets = cms.InputTag("hltPFPuppiJetForBtagEta2p4"),
    MaxTag = cms.double(999999.0),
    MinJets = cms.int32(3),
    MinTag = cms.double(0.38),
    TriggerType = cms.int32(86),
    saveTags = cms.bool(True)
)
