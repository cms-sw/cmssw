import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToTauTauETauHLTFilter = copy.deepcopy(hltHighLevel)
zToTauTauETauHLTFilter.HLTPaths = ['HLT1Electron', 'HLTXElectronTau']

