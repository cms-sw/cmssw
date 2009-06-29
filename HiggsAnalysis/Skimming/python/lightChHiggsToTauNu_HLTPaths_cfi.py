import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
lightChHiggsToTauNuHLTFilter = copy.deepcopy(hltHighLevel)

#Trigger Paths for 3_1_X
lightChHiggsToTauNuHLTFilter.HLTPaths = [ 'HLT_Mu9', 'HLT_Ele15_LW_L1R' ]
