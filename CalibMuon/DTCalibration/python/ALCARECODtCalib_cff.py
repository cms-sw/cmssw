import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

# AlCaReco for DT calibration
ALCARECODtCalibHLTFilter = copy.deepcopy(hltHighLevel)
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits
#ALCARECODtCalibHLTFilter.HLTPaths = ['HLT_L1MuOpen*', 'HLT_L1Mu*']
ALCARECODtCalibHLTFilter.throw = False ## dont throw on unknown path names
ALCARECODtCalibHLTFilter.eventSetupPathsKey = 'MuAlcaDtCalibMu'

import RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_LinearDriftFromDB_cfi as dt4DSegmentsCfiRef
dt4DSegmentsNoWire = dt4DSegmentsCfiRef.dt4DSegments.clone()
dt4DSegmentsNoWire.Reco4DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False
dt4DSegmentsNoWire.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.tTrigModeConfig.doWirePropCorrection = False

#this is to select collisions
primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

noscraping = cms.EDFilter("FilterOutScraping",
   applyfilter = cms.untracked.bool(True),
   debugOn = cms.untracked.bool(False),
   numtrack = cms.untracked.uint32(10),
   thresh = cms.untracked.double(0.25)
)

seqALCARECODtCalib = cms.Sequence(primaryVertexFilter * noscraping * ALCARECODtCalibHLTFilter * DTCalibMuonSelection * dt4DSegmentsNoWire) 
