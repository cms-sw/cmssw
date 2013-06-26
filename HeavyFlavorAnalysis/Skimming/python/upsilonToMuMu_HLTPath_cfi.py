import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
upsilonToMuMuHLTFilter = copy.deepcopy(hltHighLevel)
upsilonToMuMuHLTFilter.HLTPaths = ['HLT_DoubleMu3_Upsilon']

