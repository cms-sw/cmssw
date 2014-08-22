import FWCore.ParameterSet.Config as cms

bTagValidation     = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
   TriggerResults  = cms.InputTag('TriggerResults'),

#2012
#   HLTPathName     = cms.string('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),
   HLTPathNames     = cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),
   L25IPTagInfo    = cms.VInputTag(cms.InputTag('hltBLifetimeL25TagInfosbbPhiL1FastJetFastPV')),
   L25JetTag       = cms.VInputTag(cms.InputTag('hltBLifetimeL25BJetTagsbbPhiL1FastJetFastPV')),
   L3IPTagInfo     = cms.VInputTag(cms.InputTag('hltBLifetimeL3TagInfosbbPhiL1FastJetFastPV')),
   L3JetTag        = cms.VInputTag(cms.InputTag('hltBLifetimeL3BJetTagsbbPhiL1FastJetFastPV')),
###
   TrackIPTagInfo  = cms.InputTag('impactParameterTagInfos'),
   OfflineJetTag   = cms.InputTag('trackCountingHighEffBJetTags'),
   MinJetPT        = cms.double(20),
#   IsData          = cms.bool(False),
   BTagAlgorithms   = cms.vstring('TCHE'),
# MC stuff
   mcFlavours = cms.PSet(
      light = cms.vuint32(1, 2, 3, 21),   # udsg
      c = cms.vuint32(4),
      b = cms.vuint32(5),
      g = cms.vuint32(21),
      uds = cms.vuint32(1, 2, 3)
    ),
   mcPartons = cms.InputTag("hltJetsbyValAlgo")    # pick hltJetsbyValPhys or hltJetsbyValAlgo
)


