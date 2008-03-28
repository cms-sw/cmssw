import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
l1RPCHLTFilter = copy.deepcopy(hltHighLevel)
l1RPCHLTFilter.HLTPaths = ['CandHLT1MuonLevel1']

