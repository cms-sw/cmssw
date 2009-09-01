import FWCore.ParameterSet.Config as cms

hltMonBTagMuSource = cms.EDFilter('HLTMonBTagMuSource',
    monitorName = cms.string('HLT/HLTMonBJet'),
    pathName    = cms.string('HLT_BTagMu_Jet20'),
    L2Jets      = cms.InputTag('hltMCJetCorJetIcone5'),
    L25TagInfo  = cms.InputTag('hltBSoftMuonL25TagInfos'),
    L25JetTags  = cms.InputTag('hltBSoftMuonL25BJetTagsByDR'),
    L3TagInfo   = cms.InputTag('hltBSoftMuonL3TagInfos'),
    L3JetTags   = cms.InputTag('hltBSoftMuonL3BJetTagsByDR'),
    outputFile  = cms.untracked.string('HLTMonBTag.root'),
    storeROOT   = cms.untracked.bool(False)
)
