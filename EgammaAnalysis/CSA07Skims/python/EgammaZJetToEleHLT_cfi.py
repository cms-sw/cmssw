import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaZJetToEleHLT = copy.deepcopy(hltHighLevel)
EgammaZJetToEleHLT.HLTPaths = ['HLT2Electron', 'HLT2ElectronRelaxed']

