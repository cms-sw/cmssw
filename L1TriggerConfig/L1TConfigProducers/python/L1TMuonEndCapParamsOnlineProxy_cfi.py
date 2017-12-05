import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuonEndcap.fakeEmtfParams_cff import *

L1TMuonEndCapParamsOnlineProxy = cms.ESProducer("L1TMuonEndCapParamsOnlineProxy",
    PtAssignVersion = cms.untracked.uint32(1),
    firmwareVersion = cms.untracked.uint32(47423),
    changeDate      = cms.untracked.uint32(20161101)
)
