import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECORpcCalHLTFilter = copy.deepcopy(hltHighLevel)
seqALCARECORpcCalHLT = cms.Sequence(ALCARECORpcCalHLTFilter)
ALCARECORpcCalHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECORpcCalHLTFilter.throw = False## dont throw except on unknown path name

ALCARECORpcCalHLTFilter.HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_Mu3", "HLT_Mu5", "HLT_Mu7", "HLT_Mu9", "HLT_Mu11", "HLT_Mu13", "HLT_Mu15", "HLT_L2Mu9", "HLT_IsoMu9", "HLT_IsoMu11", "HLT_IsoMu13", "HLT_IsoMu15"]

