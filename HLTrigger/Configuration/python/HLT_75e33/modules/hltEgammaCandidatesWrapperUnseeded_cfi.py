import FWCore.ParameterSet.Config as cms

hltEgammaCandidatesWrapperUnseeded = cms.EDFilter("HLTEgammaTriggerFilterObjectWrapper",
    candIsolatedTag = cms.InputTag("hltEgammaCandidatesUnseeded"),
    candNonIsolatedTag = cms.InputTag(""),
    doIsolated = cms.bool(True),
    saveTags = cms.bool(True)
)
# foo bar baz
# 4G5gnwrMMeINq
# XerznWNpR1cTE
