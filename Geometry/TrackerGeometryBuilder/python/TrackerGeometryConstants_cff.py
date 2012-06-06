import FWCore.ParameterSet.Config as cms

TrackerGeometryService = cms.Service("TrackerGeometryService",
                                     upgradeGeometry = cms.untracked.bool(False),
                                     ROWS_PER_ROC = cms.untracked.int32(80),
                                     COLS_PER_ROC = cms.untracked.int32(52),
                                     BIG_PIX_PER_ROC_X = cms.untracked.int32(1),
                                     BIG_PIX_PER_ROC_Y = cms.untracked.int32(2),
                                     ROCS_X = cms.untracked.int32(0),
                                     ROCS_Y = cms.untracked.int32(0)
                                     )
