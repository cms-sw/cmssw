import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaZJetToMuHLT = copy.deepcopy(hltHighLevel)
EgammaZJetToMuHLT.HLTPaths = ['HLT_DoubleMu3']

