import FWCore.ParameterSet.Config as cms

PotentialTIBTECFrameHeaderEventsAdditionalPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                 commonConfiguration = cms.untracked.PSet(
                                                    historyProduct             = cms.untracked.InputTag("consecutiveHEs"),
                                                    APVPhaseLabel              = cms.untracked.string("APVPhases"),
                                                 ),
                                                 filterConfigurations = cms.untracked.VPSet(
                                                   cms.PSet(
                                                     apvModes                   = cms.untracked.vint32(47),
                                                     partitionName              = cms.untracked.string("Any"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,25)
                                                   ),
                                                   cms.PSet(
                                                     apvModes                   = cms.untracked.vint32(37),
                                                     partitionName              = cms.untracked.string("Any"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(22,23)
                                                   )
                                                  ) 
                                                 )
