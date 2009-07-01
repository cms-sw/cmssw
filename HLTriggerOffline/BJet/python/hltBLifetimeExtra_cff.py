import FWCore.ParameterSet.Config as cms

import HLTrigger.btau.jetTag.hltJetTag_cfi
# from ConfDB
hltBLifetimeL3filter = HLTrigger.btau.jetTag.hltJetTag_cfi.hltJetTag.clone()
import HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi
hltBLifetimeHLTJets = HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi.getJetsFromHLTobject.clone()
import HLTrigger.btau.jetTag.hltJetTag_cfi
# from ConfDB (relaxed triggers)
hltBLifetimeL3filterRelaxed = HLTrigger.btau.jetTag.hltJetTag_cfi.hltJetTag.clone()
import HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi
hltBLifetimeHLTJetsRelaxed = HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi.getJetsFromHLTobject.clone()
# (re)generate some productes used by the validation path
# only run on L3 events
require_hltBLifetimeL3BJetTags = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("hltBLifetimeL3BJetTags")
)

# same, for relaxed triggers
# only run on L3 events (relaxed triggers)
require_hltBLifetimeL3BJetTagsRelaxed = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("hltBLifetimeL3BJetTagsRelaxed")
)

hltBLifetimeExtra = cms.Sequence(require_hltBLifetimeL3BJetTags*hltBLifetimeL3filter*hltBLifetimeHLTJets)
hltBLifetimeExtraRelaxed = cms.Sequence(require_hltBLifetimeL3BJetTagsRelaxed*hltBLifetimeL3filterRelaxed*hltBLifetimeHLTJetsRelaxed)
hltBLifetimeL3filter.JetTag = 'hltBLifetimeL3BJetTags'
hltBLifetimeL3filter.MinTag = 6
hltBLifetimeHLTJets.jets = 'hltBLifetimeL3filter'
hltBLifetimeL3filterRelaxed.JetTag = 'hltBLifetimeL3BJetTagsRelaxed'
hltBLifetimeL3filterRelaxed.MinTag = 3.5
hltBLifetimeHLTJetsRelaxed.jets = 'hltBLifetimeL3filterRelaxed'

