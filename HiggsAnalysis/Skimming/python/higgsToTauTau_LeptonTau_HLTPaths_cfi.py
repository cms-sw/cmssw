import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauLeptonTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauLeptonTauHLTFilter.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_Mu15', 'HLT_IsoMu11']

