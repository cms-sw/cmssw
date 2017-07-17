import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.GeometryDB_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff import *
from CondCore.DBCommon.CondDBSetup_cfi import *

from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
dt2DSegments.Reco2DAlgoConfig.performT0_vdriftSegCorrection = True
dt4DSegments.Reco4DAlgoConfig.performT0_vdriftSegCorrection = True

from EventFilter.DTRawToDigi.dtunpacker_cfi import *

dtCalibOfflineReco = cms.Sequence(dt1DRecHits + dt2DSegments + dt4DSegments)
dtCalibOfflineRecoRAW = cms.Sequence(muonDTDigis + dt1DRecHits + dt2DSegments + dt4DSegments)
