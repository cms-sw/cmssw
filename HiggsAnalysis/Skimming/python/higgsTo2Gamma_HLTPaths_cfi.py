import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsTo2GammaHLTFilter = copy.deepcopy(hltHighLevel)
higgsTo2GammaHLTFilter.HLTPaths = ['HLT1Photon', 'HLT1PhotonRelaxed', 'HLT2Photon', 'HLT2PhotonRelaxed']

