import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
heavyChHiggsToTauNuHLTFilter = copy.deepcopy(hltHighLevel)
heavyChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_IsoTau_MET65_Trk20']

