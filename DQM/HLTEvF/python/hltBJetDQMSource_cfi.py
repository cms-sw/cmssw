import FWCore.ParameterSet.Config as cms

hltBJetDQMSource = cms.EDFilter("HLTBJetDQMSource",
    softmuonL3JetTags = cms.InputTag("hltBSoftmuonL3BJetTags"),
    softmuonL25TagInfo = cms.InputTag("hltBSoftmuonL25TagInfos"),
    softmuonL3TagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    performanceL25JetTags = cms.InputTag("hltBSoftmuonL25BJetTags"),
    lifetimeL3JetTags = cms.InputTag("hltBLifetimeL3BJetTags"),
    performanceL3JetTags = cms.InputTag("hltBSoftmuonL3BJetTagsByDR"),
    monitorName = cms.untracked.string('HLT/HLTMonBJet'),
    storeROOT = cms.untracked.bool(False),
    performanceL3TagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    lifetimeL2Jets = cms.InputTag("hltMCJetCorJetIcone5"),
    outputFile = cms.untracked.string('HLTBJetDQM.root'),
    performanceL25TagInfo = cms.InputTag("hltBSoftmuonL25TagInfos"),
    softmuonL25JetTags = cms.InputTag("hltBSoftmuonL25BJetTags"),
    performanceL2Jets = cms.InputTag("hltMCJetCorJetIcone5"),
    lifetimeL25JetTags = cms.InputTag("hltBLifetimeL25BJetTags"),
    softmuonL2Jets = cms.InputTag("hltMCJetCorJetIcone5"),
    lifetimeL25TagInfo = cms.InputTag("hltBLifetimeL25TagInfos"),
    lifetimeL3TagInfo = cms.InputTag("hltBLifetimeL3TagInfos")
)


