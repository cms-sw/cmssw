import FWCore.ParameterSet.Config as cms

hltBTagPFPuppiDeepFlavour0p935DoubleEta2p4 = cms.EDFilter("HLTPFJetTag",
    JetTags = cms.InputTag("hltPfDeepFlavourJetTagsModEta2p4","probb"),
    Jets = cms.InputTag("hltPFPuppiJetForBtagEta2p4"),
    MaxTag = cms.double(999999.0),
    MinJets = cms.int32(2),
    MinTag = cms.double(0.935),
    TriggerType = cms.int32(86),
    MatchJetsByDeltaR = cms.bool(True),
    MaxJetDeltaR = cms.double(0.1),
    saveTags = cms.bool(True)
)
