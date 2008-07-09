import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
hltBLifetimeL3filter = copy.deepcopy(hltJetTag)
import copy
from HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi import *
hltBLifetimeL3Jets = copy.deepcopy(getJetsFromHLTobject)
hltBLifetimeL3reco = cms.Sequence(hltBLifetimeL3Jets+cms.SequencePlaceholder("hltBLifetimeL3tracking")*cms.SequencePlaceholder("hltBLifetimeL3Associator")*cms.SequencePlaceholder("hltBLifetimeL3TagInfos")*cms.SequencePlaceholder("hltBLifetimeL3BJetTags"))
hltBLifetimeL3filter.JetTag = 'hltBLifetimeL3BJetTags'
hltBLifetimeL3filter.MinTag = 6

