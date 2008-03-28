import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
wToTauNuHLTFilter = copy.deepcopy(hltHighLevel)
wToTauNuHLTFilter.HLTPaths = ['HLT1Tau1MET']

