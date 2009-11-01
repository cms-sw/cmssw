import FWCore.ParameterSet.Config as cms

PotentialTIBTECFrameHeaderEventsFPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                 commonConfiguration = cms.untracked.PSet(
                                                    historyProduct             = cms.untracked.InputTag("ConsecutiveHEs"),
                                                    APVPhaseLabel              = cms.untracked.string("apvPhases"),
                                                 ),
                                                 filterConfigurations = cms.untracked.VPSet(
                                                     cms.PSet(
                                                     partitionName              = cms.untracked.string("Any"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,22)
                                                    )
                                                   ) 
                                                 )
