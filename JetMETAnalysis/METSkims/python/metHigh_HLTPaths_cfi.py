import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
metHLTFilter = copy.deepcopy(hltHighLevel)
metHLTFilter.HLTPaths = ['HLT1MET']

