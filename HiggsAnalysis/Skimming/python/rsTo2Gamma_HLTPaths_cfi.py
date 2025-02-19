import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
rsTo2GammaHLTFilter = copy.deepcopy(hltHighLevel)
rsTo2GammaHLTFilter.HLTPaths = ['HLT_EM80', 'HLT_EM200']

