import FWCore.ParameterSet.Config as cms

PotentialTIBTECHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                               commonConfiguration = cms.untracked.PSet(
                                 historyProduct             = cms.untracked.InputTag("consecutiveHEs"),
                                 APVPhaseLabel              = cms.untracked.string("APVPhases"),
                               ),
                               filterConfigurations = cms.untracked.VPSet(
                                 cms.PSet(
                                    apvModes                   = cms.untracked.vint32(47),
                                    partitionName              = cms.untracked.string("Any"),
                                    absBXInCycleRangeLtcyAware = cms.untracked.vint32(8,8)
                                 ),
                                 cms.PSet(
                                    apvModes                   = cms.untracked.vint32(37),
                                    partitionName              = cms.untracked.string("Any"),
                                    absBXInCycleRangeLtcyAware = cms.untracked.vint32(6,8)
                                 )
                               )
                             )
