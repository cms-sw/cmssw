import FWCore.ParameterSet.Config as cms

DigiSource = cms.EDFilter(
    "SiStripTrivialDigiSource",
    MeanOccupancy = cms.untracked.double(1.0),
    RmsOccupancy = cms.untracked.double(0.1),
    FedRawDataMode = cms.untracked.bool(False),
    UseFedKey = cms.untracked.bool(False),
    PedestalLevel = cms.untracked.int32(0),
    )
