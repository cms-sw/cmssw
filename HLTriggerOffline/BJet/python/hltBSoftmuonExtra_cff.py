import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
# from HLTrigger/btau/data/jetTag/softmuonL3.cff - b HLT modules for Level 3.
hltBSoftmuonL3filter = copy.deepcopy(hltJetTag)
import copy
from HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi import *
hltBSoftmuonHLTJets = copy.deepcopy(getJetsFromHLTobject)
import copy
from HLTrigger.btau.jetTag.hltJetTag_cfi import *
# from HLTrigger/btau/data/jetTag/softmuonL3.cff - no pTrel cut for Sys8 studies / use dR isa
hltBSoftmuonByDRL3filter = copy.deepcopy(hltJetTag)
import copy
from HLTrigger.btau.jetTag.getJetsFromHLTobject_cfi import *
hltBSoftmuonHLTJetsByDR = copy.deepcopy(getJetsFromHLTobject)
# (re)generate some productes used by the validation path
# only run on L3 events
require_hltBSoftmuonL3BJetTags = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("hltBSoftmuonL3BJetTags")
)

hltBSoftmuonExtra = cms.Sequence(require_hltBSoftmuonL3BJetTags*hltBSoftmuonL3filter*hltBSoftmuonHLTJets+hltBSoftmuonByDRL3filter*hltBSoftmuonHLTJetsByDR)
hltBSoftmuonL3filter.JetTag = 'hltBSoftmuonL3BJetTags'
hltBSoftmuonL3filter.MinTag = 0.7 ## pT_rel in GeV/c

hltBSoftmuonHLTJets.jets = 'hltBSoftmuonL3filter'
hltBSoftmuonByDRL3filter.JetTag = 'hltBSoftmuonL3BJetTagsByDR'
hltBSoftmuonByDRL3filter.MinTag = 0.5 ## 1 if dR < 0.5, 0 otherwise

hltBSoftmuonHLTJetsByDR.jets = 'hltBSoftmuonByDRL3filter'

