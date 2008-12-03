import FWCore.ParameterSet.Config as cms

DigiSource = cms.EDFilter("SiStripTrivialDigiSource",
    FedRawDataMode = cms.untracked.bool(False),
    MeanOccupancy = cms.untracked.double(1.0),
    TestDistribution = cms.untracked.bool(False),
    UseFedKey = cms.untracked.bool(False),
    RmsOccupancy = cms.untracked.double(0.1)
)


