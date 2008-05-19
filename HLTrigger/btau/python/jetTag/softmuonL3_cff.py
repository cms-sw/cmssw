import FWCore.ParameterSet.Config as cms

import HLTrigger.btau.jetTag.hltJetTag_cfi
# b HLT modules for Level 3.
hltBSoftmuonL3filter = HLTrigger.btau.jetTag.hltJetTag_cfi.hltJetTag.clone()
import HLTrigger.btau.jetTag.hltJetTag_cfi
# no pTrel cut for Sys8 studies / use dR isa
hltBSoftmuonByDRL3filter = HLTrigger.btau.jetTag.hltJetTag_cfi.hltJetTag.clone()
# Event reconstruction needed for trigger.
hltBSoftmuonL3reco = cms.Sequence(cms.SequencePlaceholder("l3muonrecoNocand")*cms.SequencePlaceholder("hltBSoftmuonL3TagInfos")*cms.SequencePlaceholder("hltBSoftmuonL3BJetTags")*cms.SequencePlaceholder("hltBSoftmuonL3BJetTagsByDR"))
hltBSoftmuonL3filter.JetTag = 'hltBSoftmuonL3BJetTags'
hltBSoftmuonL3filter.MinTag = 0.7 ## pT_rel in GeV/c

hltBSoftmuonL3filter.SaveTag = True
hltBSoftmuonByDRL3filter.JetTag = 'hltBSoftmuonL3BJetTagsByDR'
hltBSoftmuonByDRL3filter.MinTag = 0.5 ## 1 if dR < 0.5, 0 otherwise

hltBSoftmuonByDRL3filter.SaveTag = True

