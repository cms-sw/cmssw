import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToEEHLTFilter = copy.deepcopy(hltHighLevel)
zToEEHLTFilter.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed', 'HLT2Electron', 'HLT2ElectronRelaxed', 'HLTXElectronMuon']

