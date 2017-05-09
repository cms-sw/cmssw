import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibCosmicsHLTFilter = copy.deepcopy(hltHighLevel)
ALCARECODtCalibCosmicsHLTFilter.throw = False ## dont throw on unknown path names
ALCARECODtCalibCosmicsHLTFilter.eventSetupPathsKey = 'DtCalibCosmics'
#ALCARECODtCalibCosmicsHLTFilter.HLTPaths = ['HLT_L1SingleMuOpen_AntiBPTX_v*','HLT_L1TrackerCosmics_v*']
#ALCARECODtCalibCosmicsHLTFilter.eventSetupPathsKey = ''

#import RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_LinearDriftFromDB_cfi as dt4DSegmentsCfiRef
#dt4DSegmentsNoWire = dt4DSegmentsCfiRef.dt4DSegments.clone()
#dt4DSegmentsNoWire.Reco4DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False
#dt4DSegmentsNoWire.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False

#seqALCARECODtCalibCosmics = cms.Sequence(ALCARECODtCalibCosmicsHLTFilter * dt4DSegmentsNoWire)
seqALCARECODtCalibCosmics = cms.Sequence(ALCARECODtCalibCosmicsHLTFilter)
