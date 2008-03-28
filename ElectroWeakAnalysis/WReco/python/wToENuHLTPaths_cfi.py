import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
wToENuHLTFilter = copy.deepcopy(hltHighLevel)
wToENuHLTFilter.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed']

