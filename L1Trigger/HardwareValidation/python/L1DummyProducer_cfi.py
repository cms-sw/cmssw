import FWCore.ParameterSet.Config as cms

l1dummy = cms.EDProducer("L1DummyProducer",
    DO_SYSTEM = cms.untracked.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0),
    EnergyBase = cms.untracked.double(100.0),
    VerboseFlag = cms.untracked.int32(0),
    EnergySigm = cms.untracked.double(10.0)
)


