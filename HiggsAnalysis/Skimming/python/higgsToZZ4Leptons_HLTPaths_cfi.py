import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToZZ4LeptonsHLTFilter = copy.deepcopy(hltHighLevel)
higgsToZZ4LeptonsHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLT1Electron', 'HLT1ElectronRelaxed', 
    'HLT2Electron', 'HLT2ElectronRelaxed']

