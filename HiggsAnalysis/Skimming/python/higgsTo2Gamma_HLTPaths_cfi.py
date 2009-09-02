import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsTo2GammaHLTFilter = copy.deepcopy(hltHighLevel)
higgsTo2GammaHLTFilter.HLTPaths = ['HLT_IsoPhoton30_L1I', 'HLT_IsoPhoton40_L1R', 'HLT_DoubleIsoPhoton20_L1I', 'HLT_DoubleIsoPhoton20_L1R']

