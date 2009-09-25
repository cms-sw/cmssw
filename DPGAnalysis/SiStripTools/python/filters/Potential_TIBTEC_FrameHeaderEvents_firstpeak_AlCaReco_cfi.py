import FWCore.ParameterSet.Config as cms

PotentialTIBTECFrameHeaderEventsFPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                 commonConfiguration = cms.untracked.PSet(
                                                    historyProduct             = cms.untracked.InputTag("ConsecutiveHEs"),
                                                    APVPhaseLabel              = cms.untracked.string("apvPhases"),
                                                     partitionName              = cms.untracked.string("TM")
                                                 ),
                                                   filterConfigurations = cms.untracked.VPSet(
                                                    cms.PSet(
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,22)
                                                    )
                                                   ) 
                                                 )
