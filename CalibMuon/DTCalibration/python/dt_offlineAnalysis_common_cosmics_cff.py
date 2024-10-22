import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.GeometryDB_cff import *
from Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from CondCore.CondDB.CondDB_cfi import *

from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *

from EventFilter.DTRawToDigi.dtunpacker_cfi import *

dtCalibOfflineReco = cms.Sequence(dt1DRecHits + dt2DSegments + dt4DSegments)
dtCalibOfflineRecoRAW = cms.Sequence(muonDTDigis + dt1DRecHits + dt2DSegments + dt4DSegments)
