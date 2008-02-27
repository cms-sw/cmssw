import FWCore.ParameterSet.Config as cms

EDMtoMEConverter = cms.EDFilter("EDMtoMEConverter",
    Frequency = cms.untracked.int32(50),
    Verbosity = cms.untracked.int32(0),
    Outputfile = cms.string('EDMtoMEConverter.root'),
    Name = cms.untracked.string('EDMtoMEConverter')
)




