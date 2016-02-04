import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
bToMuMuHLTFilter = copy.deepcopy(hltHighLevel)
bToMuMuHLTFilter.HLTPaths = ['HLT_DoubleMu3', 'HLT_DoubleMu4_BJPsi', 'HLT_DoubleMu3_SameSign']

