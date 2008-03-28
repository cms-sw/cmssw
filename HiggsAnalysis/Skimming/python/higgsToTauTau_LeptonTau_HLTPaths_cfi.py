import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauLeptonTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauLeptonTauHLTFilter.HLTPaths = ['HLT1Electron', 'HLTXElectronTau', 'HLT1MuonIso', 'HLTXMuonTau']

