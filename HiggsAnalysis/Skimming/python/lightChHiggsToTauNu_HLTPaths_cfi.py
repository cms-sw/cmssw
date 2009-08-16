import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
lightChHiggsToTauNuHLTFilter = copy.deepcopy(hltHighLevel)
lightChHiggsToTauNuHLTFilter.TriggerResultsTag  = cms.InputTag("TriggerResults::HLT")

#Lum 8E29
lightChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_Mu3','HLT_Ele10_LW_L1R']

#Lum 1E31
#lightChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_Mu15','HLT_Ele15_SW_LooseTrackIso_L1R']
