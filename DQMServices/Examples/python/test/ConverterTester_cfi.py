import FWCore.ParameterSet.Config as cms

convertertester = cms.EDFilter("ConverterTester",
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    Name = cms.untracked.string('ConverterTester')
)

