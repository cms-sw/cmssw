import FWCore.ParameterSet.Config as cms

hltMonBTagMuSource = cms.EDAnalyzer('HLTMonBTagMuSource',
    triggerResults  = cms.InputTag('TriggerResults', '', 'HLT'),
    monitorName     = cms.string('HLT/HLTMonBJet'),
    processName     = cms.string('HLT'),
    pathName        = cms.string('HLT_BTagMu_Jet10U'),
    interestingJets = cms.uint32( 4 ),
    L1Filter        = cms.InputTag('hltL1sBTagMuJet10U'),
    L2Filter        = cms.InputTag('hltBJet10U'),
    L2Jets          = cms.InputTag('hltMCJetCorJetIcone5HF07'),
    L25TagInfo      = cms.InputTag('hltBSoftMuonL25TagInfosU'),
    L25JetTags      = cms.InputTag('hltBSoftMuonL25BJetTagsUByDR'),
    L25Filter       = cms.InputTag('hltBSoftMuonL25FilterUByDR'),
    L3TagInfo       = cms.InputTag('hltBSoftMuonL3TagInfosU'),
    L3JetTags       = cms.InputTag('hltBSoftMuonL3BJetTagsUByDR'),
    L3Filter        = cms.InputTag('hltBSoftMuonL3FilterUByDR'),
    storeROOT       = cms.untracked.bool(False),
    outputFile      = cms.untracked.string('HLTMonBTag.root')
)
