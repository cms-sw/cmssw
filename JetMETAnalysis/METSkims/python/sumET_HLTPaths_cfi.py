import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
sumETHLTFilter = copy.deepcopy(hltHighLevel)
sumETHLTFilter.HLTPaths = ['CandHLT1SumET']

