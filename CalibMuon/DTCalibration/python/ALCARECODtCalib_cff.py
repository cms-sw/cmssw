import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibHLTFilter = copy.deepcopy(hltHighLevel)

ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECODtCalibHLTFilter.throw = False ## dont throw on unknown path names

ALCARECODtCalibHLTFilter.HLTPaths = ['HLT_L1MuOpen', 'HLT_L1Mu']

seqALCARECODtCalib = cms.Sequence(ALCARECODtCalibHLTFilter*DTCalibMuonSelection) 
