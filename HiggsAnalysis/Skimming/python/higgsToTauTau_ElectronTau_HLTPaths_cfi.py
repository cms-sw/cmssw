import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauElectronTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauElectronTauHLTFilter.TriggerResultsTag  = cms.InputTag("TriggerResults","","HLT8E29")

#higgsToTauTauElectronTauHLTFilter.HLTPaths = ['HLT_IsoEle15_L1I']
higgsToTauTauElectronTauHLTFilter.HLTPaths = ['HLT_Ele10_LW_EleId_L1R']

