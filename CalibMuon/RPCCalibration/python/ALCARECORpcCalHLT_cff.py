import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECORpcCalHLTFilter = copy.deepcopy(hltHighLevel)
seqALCARECORpcCalHLT = cms.Sequence(ALCARECORpcCalHLTFilter)
ALCARECORpcCalHLTFilter.andOr = True ## choose logical OR between Triggerbits

ALCARECORpcCalHLTFilter.HLTPaths = ['HLT1MuonLevel1']

