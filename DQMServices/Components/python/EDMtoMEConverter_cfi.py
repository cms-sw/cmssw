import FWCore.ParameterSet.Config as cms

EDMtoMEConverter = cms.EDFilter("EDMtoMEConverter",
    Frequency = cms.untracked.int32(50), ## frequency of current processing blurb

    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    # 1 provides basic output
    Outputfile = cms.string('EDMtoMEConverter.root'),
    Name = cms.untracked.string('EDMtoMEConverter')
)




