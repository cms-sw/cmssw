import FWCore.ParameterSet.Config as cms

hltMonBTagIPSource = cms.EDFilter('HLTMonBTagIPSource',
    triggerResults  = cms.InputTag('TriggerResults', '', 'HLT'),
    monitorName     = cms.string('HLT/HLTMonBJet'),
    processName     = cms.string('HLT'),
    pathName        = cms.string('HLT_BTagIP_Jet50U'),
    interestingJets = cms.uint32( 4 ),
    L1Filter        = cms.InputTag('hltL1sBTagIPJet50U'),
    L2Filter        = cms.InputTag('hltBJet50U'),
    L2Jets          = cms.InputTag('hltMCJetCorJetIcone5HF07'),
    L25TagInfo      = cms.InputTag('hltBLifetimeL25TagInfosStartupU'),
    L25JetTags      = cms.InputTag('hltBLifetimeL25BJetTagsStartupU'),
    L25Filter       = cms.InputTag('hltBLifetimeL25FilterStartupU'),
    L3TagInfo       = cms.InputTag('hltBLifetimeL3TagInfosStartupU'),
    L3JetTags       = cms.InputTag('hltBLifetimeL3BJetTagsStartupU'),
    L3Filter        = cms.InputTag('hltBLifetimeL3FilterStartupU'),
    storeROOT       = cms.untracked.bool(False),
    outputFile      = cms.untracked.string('HLTMonBTag.root')
)
