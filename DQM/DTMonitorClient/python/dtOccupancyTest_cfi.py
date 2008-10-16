import FWCore.ParameterSet.Config as cms

dtOccupancyTest = cms.EDAnalyzer("DTOccupancyTest",
                                 testPulseMode = cms.untracked.bool(False))


