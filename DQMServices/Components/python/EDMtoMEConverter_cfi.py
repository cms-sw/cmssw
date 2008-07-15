import FWCore.ParameterSet.Config as cms

EDMtoMEConverter = cms.EDFilter("EDMtoMEConverter",
    Name = cms.untracked.string('EDMtoMEConverter'),
    Verbosity = cms.untracked.int32(0), # 0 provides no output
                                        # 1 provides basic output
    Frequency = cms.untracked.int32(50),
    convertOnEndLumi = cms.untracked.bool(False),
    convertOnEndRun = cms.untracked.bool(False)
)

