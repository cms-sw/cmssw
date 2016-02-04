import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaVeryHighEtTrigger = copy.deepcopy(hltHighLevel)
EgammaVeryHighEtTrigger.HLTPaths = ['HLT_EM200']

