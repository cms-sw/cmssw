import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsTo2GammaHLTFilter = copy.deepcopy(hltHighLevel)
#This is for 2_2_X
#higgsTo2GammaHLTFilter.HLTPaths = ['HLT_IsoPhoton30_L1I', 'HLT_IsoPhoton40_L1R', 'HLT_DoubleIsoPhoton20_L1I', 'HLT_DoubleIsoPhoton20_L1R','HLT_Photon25_L1R','HLT_Photon15_L1R']
#This is for 3_1_X
higgsTo2GammaHLTFilter.HLTPaths = ['HLT_Photon15_L1R', 'HLT_DoublePhoton10_L1R', 'HLT_Photon20_L1R']
