import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToTauTau_EMuHLTFilter = copy.deepcopy(hltHighLevel)
zToTauTau_EMuHLTFilter.HLTPaths = ['HLTXElectronMuon', 'HLT1MuonIso', 'HLT1Electron']

