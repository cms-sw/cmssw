import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaWJetToEleHLT = copy.deepcopy(hltHighLevel)
EgammaWJetToEleHLT.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed']

