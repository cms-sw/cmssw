import FWCore.ParameterSet.Config as cms

import HLTrigger.btau.jetTag.hltJetTag_cfi
# b HLT modules for Level 3.
hltBLifetimeL3filter = HLTrigger.btau.jetTag.hltJetTag_cfi.hltJetTag.clone()
import HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi
hltBLifetimeL3Jets = HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi.getJetsFromHLTobject.clone()
# Reconstruction sequences needed before each trigger.
hltBLifetimeL3reco = cms.Sequence(hltBLifetimeL3Jets+cms.SequencePlaceholder("hltBLifetimeL3tracking")*cms.SequencePlaceholder("hltBLifetimeL3Associator")*cms.SequencePlaceholder("hltBLifetimeL3TagInfos")*cms.SequencePlaceholder("hltBLifetimeL3BJetTags"))
hltBLifetimeL3filter.JetTag = 'hltBLifetimeL3BJetTags'
hltBLifetimeL3filter.MinTag = 6 ## best for trackCounting, 2nd track

#replace hltBLifetimeL3filter.MinTag = 5    # best for trackCounting, 3rd track
hltBLifetimeL3filter.SaveTag = True

