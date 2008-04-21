import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToWW2LeptonsHLTFilter = copy.deepcopy(hltHighLevel)
higgsToWW2LeptonsHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT1Electron', 'HLT1ElectronRelaxed', 'HLT2MuonNonIso', 
    'HLT2Electron', 'HLT2ElectronRelaxed', 'HLTXElectronMuon', 'HLTXElectronMuonRelaxed']

