import FWCore.ParameterSet.Config as cms

EDMtoMEConverter = cms.EDFilter("EDMtoMEConverter",
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    # 1 provides basic output
    Frequency = cms.untracked.int32(50),
    Name = cms.untracked.string('EDMtoMEConverter')
)


