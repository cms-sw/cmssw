import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("TestRunLumiSource",
    runLumiEvent = cms.untracked.vint32(1, 0, 0,
                                        1, 1, 0,
                                        1, 1, 1,
                                        1, 1, 2,
                                        1, 1, 0,
                                        1, 0, 0
    ),
    whenToThrow = cms.untracked.int32(3)
)
