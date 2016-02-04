import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauLeptonTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauLeptonTauHLTFilter.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_IsoEle12_IsoTau_Trk3', 'HLT_IsoMu11', 'HLT_IsoMu14_IsoTau_Trk3']

