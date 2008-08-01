import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
# from HLTrigger/btau/data/jetTag/lifetimeL3.cff
hltBLifetimeL3filter = copy.deepcopy(hltJetTag)
import copy
from HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi import *
hltBLifetimeHLTJets = copy.deepcopy(getJetsFromHLTobject)
# (re)generate some productes used by the validation path
# only run on L3 events
require_hltBLifetimeL3BJetTags = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("hltBLifetimeL3BJetTags")
)

hltBLifetimeExtra = cms.Sequence(require_hltBLifetimeL3BJetTags*hltBLifetimeL3filter*hltBLifetimeHLTJets)
hltBLifetimeL3filter.JetTag = 'hltBLifetimeL3BJetTags'
hltBLifetimeL3filter.MinTag = 6
hltBLifetimeHLTJets.jets = 'hltBLifetimeL3filter'

