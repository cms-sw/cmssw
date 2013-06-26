import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
jpsiToMuMuHLTFilter = copy.deepcopy(hltHighLevel)
jpsiToMuMuHLTFilter.HLTPaths = ['HLT_DoubleMu3_JPsi']

