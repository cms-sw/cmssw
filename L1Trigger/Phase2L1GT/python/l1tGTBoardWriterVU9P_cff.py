import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTBoardWriter_cff import BoardDataInput, BoardDataOutputObjects, AlgoBitBoardData


BoardDataInput.platform = cms.string("VU9P")
BoardDataOutputObjects.platform = cms.string("VU9P")
AlgoBitBoardData.channels = cms.vuint32(32, 33)
