import FWCore.ParameterSet.Config as cms

hltMonBTagIPSource = cms.EDFilter('HLTMonBTagIPSource',
    monitorName = cms.string('HLT/HLTMonBJet'),
    pathName    = cms.string('HLT_BTagIP_Jet80'),
    L2Jets      = cms.InputTag('hltMCJetCorJetIcone5Regional'),
    L25TagInfo  = cms.InputTag('hltBLifetimeL25TagInfosStartup'),
    L25JetTags  = cms.InputTag('hltBLifetimeL25BJetTagsStartup'),
    L3TagInfo   = cms.InputTag('hltBLifetimeL3TagInfosStartup'),
    L3JetTags   = cms.InputTag('hltBLifetimeL3BJetTagsStartup'),
    outputFile  = cms.untracked.string('HLTMonBTag.root'),
    storeROOT   = cms.untracked.bool(False)
)
