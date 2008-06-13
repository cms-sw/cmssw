import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
singlePhotonHLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
singleRelaxedPhotonHLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
singlePhotonHLTFilter12 = copy.deepcopy(hltHighLevel)
singlePhotonHLTFilter.HLTPaths = ['HLT_IsoPhoton30_L1I']
singleRelaxedPhotonHLTFilter.HLTPaths = ['HLT_IsoPhoton40_L1R']
singlePhotonHLTFilter12.HLTPaths = ['CandHLT1PhotonL1Isolated']

