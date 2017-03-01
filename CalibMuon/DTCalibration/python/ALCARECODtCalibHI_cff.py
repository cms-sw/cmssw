import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibHIHLTFilter = copy.deepcopy(hltHighLevel)
#ALCARECODtCalibHIHLTFilter.andOr = True ## choose logical OR between Triggerbits
#ALCARECODtCalibHIHLTFilter.HLTPaths = ['HLT_HIL1SingleMu3']
#ALCARECODtCalibHIHLTFilter.HLTPaths = ['HLT_.*']
ALCARECODtCalibHIHLTFilter.throw = False ## dont throw on unknown path names
ALCARECODtCalibHIHLTFilter.eventSetupPathsKey = 'MuAlcaDtCalibHI'

import RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_LinearDriftFromDB_cfi as dt4DSegmentsCfiRef
dt4DSegmentsNoWire = dt4DSegmentsCfiRef.dt4DSegments.clone()
dt4DSegmentsNoWire.Reco4DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False
dt4DSegmentsNoWire.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False

#seqALCARECODtCalibHI = cms.Sequence(ALCARECODtCalibHIHLTFilter * primaryVertexFilter * DTCalibMuonSelection * dt4DSegmentsNoWire) 

seqALCARECODtCalibHI = cms.Sequence(ALCARECODtCalibHIHLTFilter * dt4DSegmentsNoWire) 
