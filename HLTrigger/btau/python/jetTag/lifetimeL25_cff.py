import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
hltBLifetimeL25filter = copy.deepcopy(hltJetTag)
hltBLifetimeL25reco = cms.Sequence(cms.SequencePlaceholder("hltBLifetimeL25jetselection")+cms.SequencePlaceholder("hltBLifetimeL25tracking")*cms.SequencePlaceholder("hltBLifetimeL25Associator")*cms.SequencePlaceholder("hltBLifetimeL25TagInfos")*cms.SequencePlaceholder("hltBLifetimeL25BJetTags"))
hltBLifetimeL25filter.JetTag = 'hltBLifetimeL25BJetTags'
hltBLifetimeL25filter.MinTag = 3.5

