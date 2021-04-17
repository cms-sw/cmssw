import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibHLTFilter = copy.deepcopy(hltHighLevel)
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits
#ALCARECODtCalibHLTFilter.HLTPaths = ['HLT_L1MuOpen*', 'HLT_L1Mu*']
ALCARECODtCalibHLTFilter.throw = False ## dont throw on unknown path names
ALCARECODtCalibHLTFilter.eventSetupPathsKey = 'DtCalib'

import RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_LinearDriftFromDB_cfi as dt4DSegmentsCfiRef
dt4DSegmentsNoWire = dt4DSegmentsCfiRef.dt4DSegments.clone()
dt4DSegmentsNoWire.Reco4DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False
dt4DSegmentsNoWire.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False

#this is to select collisions
from RecoMET.METFilters.metFilters_cff import primaryVertexFilter, noscraping

seqALCARECODtCalib = cms.Sequence(primaryVertexFilter * noscraping * ALCARECODtCalibHLTFilter * DTCalibMuonSelection * dt4DSegmentsNoWire) 

## customizations for the pp_on_AA eras
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ALCARECODtCalibHLTFilter,
                  eventSetupPathsKey='DtCalibHI'
)

seqALCARECODtCalibHI = cms.Sequence(ALCARECODtCalibHLTFilter * dt4DSegmentsNoWire)

#Specify to use HI sequence for the pp_on_AA eras
pp_on_AA.toReplaceWith(seqALCARECODtCalib,seqALCARECODtCalibHI)
