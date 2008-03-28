import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToTauTau_DoubleTauHLTFilter = copy.deepcopy(hltHighLevel)
zToTauTau_DoubleTauHLTFilter.HLTPaths = ['HLT2TauPixel']

