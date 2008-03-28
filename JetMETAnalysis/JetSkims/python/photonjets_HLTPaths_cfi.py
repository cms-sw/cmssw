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
singlePhotonHLTFilter.HLTPaths = ['HLT1Photon']
singleRelaxedPhotonHLTFilter.HLTPaths = ['HLT1PhotonRelaxed']
singlePhotonHLTFilter12.HLTPaths = ['CandHLT1PhotonL1Isolated']

