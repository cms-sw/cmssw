import FWCore.ParameterSet.Config as cms

l1dummy = cms.EDProducer("L1DummyProducer",
    EnergyBase = cms.untracked.double(100.0),
    EnergySigm = cms.untracked.double(10.0),
    DO_SYSTEM = cms.untracked.vuint32(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    ),
    VerboseFlag = cms.untracked.int32(0)
)


