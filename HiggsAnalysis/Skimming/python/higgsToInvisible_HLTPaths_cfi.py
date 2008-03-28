import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToInvisibleHLTFilter = copy.deepcopy(hltHighLevel)
higgsToInvisibleHLTFilter.HLTPaths = ['HLT2jetvbfMET']

