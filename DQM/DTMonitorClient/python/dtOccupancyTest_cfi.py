import FWCore.ParameterSet.Config as cms

dtOccupancyTest = cms.EDAnalyzer("DTOccupancyTest",
                                 testPulseMode = cms.untracked.bool(False),
                                 runOnAllHitsOccupancies = cms.untracked.bool(True),
                                 runOnNoiseOccupancies = cms.untracked.bool(False),
                                 runOnInTimeOccupancies = cms.untracked.bool(False),
                                 nEventsCert = cms.untracked.int32(2500)
                                 )


