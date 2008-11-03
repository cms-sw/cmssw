import FWCore.ParameterSet.Config as cms

dtTPmonitorTest = cms.EDAnalyzer("DTOccupancyTest",
                                 testPulseMode = cms.untracked.bool(True),
                                 runOnAllHitsOccupancies = cms.untracked.bool(False),
                                 runOnNoiseOccupancies = cms.untracked.bool(False),
                                 runOnInTimeOccupancies = cms.untracked.bool(True)
                                 )

