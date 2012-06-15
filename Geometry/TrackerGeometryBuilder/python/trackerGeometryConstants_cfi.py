import FWCore.ParameterSet.Config as cms

trackerGeometryConstants = cms.PSet(
   upgradeGeometry = cms.bool(False),
   ROWS_PER_ROC = cms.int32(80),
   COLS_PER_ROC = cms.int32(52),
   BIG_PIX_PER_ROC_X = cms.int32(1),
   BIG_PIX_PER_ROC_Y = cms.int32(2),
   ROCS_X = cms.int32(0),
   ROCS_Y = cms.int32(0)
)
