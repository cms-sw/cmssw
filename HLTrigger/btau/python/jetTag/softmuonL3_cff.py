import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
hltBSoftmuonL3filter = copy.deepcopy(hltJetTag)
import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
hltBSoftmuonByDRL3filter = copy.deepcopy(hltJetTag)
hltBSoftmuonL3reco = cms.Sequence(cms.SequencePlaceholder("l3muonrecoNocand")*cms.SequencePlaceholder("hltBSoftmuonL3TagInfos")*cms.SequencePlaceholder("hltBSoftmuonL3BJetTags")*cms.SequencePlaceholder("hltBSoftmuonL3BJetTagsByDR"))
hltBSoftmuonL3filter.JetTag = 'hltBSoftmuonL3BJetTags'
hltBSoftmuonL3filter.MinTag = 0.7
hltBSoftmuonByDRL3filter.JetTag = 'hltBSoftmuonL3BJetTagsByDR'
hltBSoftmuonByDRL3filter.MinTag = 0.5

