import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
hltBSoftmuonL25filter = copy.deepcopy(hltJetTag)
hltBSoftmuonL25reco = cms.Sequence(cms.SequencePlaceholder("hltBSoftmuonL25jetselection")+cms.SequencePlaceholder("l2muonrecoNocand")*cms.SequencePlaceholder("hltBSoftmuonL25TagInfos")*cms.SequencePlaceholder("hltBSoftmuonL25BJetTags"))
hltBSoftmuonL25filter.JetTag = 'hltBSoftmuonL25BJetTags'
hltBSoftmuonL25filter.MinTag = 0.5

