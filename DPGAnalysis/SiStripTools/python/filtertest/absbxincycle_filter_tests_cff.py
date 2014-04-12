import FWCore.ParameterSet.Config as cms

# simple cut: three bins, any partition

absbxincycle1 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(5,7),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )
                             
# four bins across the boundary: "Any" partition

absbxincycle2 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(68,1),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# OR of the two filter above using the VPSet mechanism

absbxincycle3 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(68,1),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(5,7),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# with one negative boundary: "Any" partition

absbxincycle4 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(-1,3),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                      )

# another with one negative boundary: "Any" partition

absbxincycle5 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(65,-1),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                      )

# OR with negative boundaries: "Any" partition

absbxincycle6 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(65,-1),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(20,20),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(-1,3),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# now a set of filters which are NOT Latency Aware

                       
absbxincycle11 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(5,7),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# four bins across the boundary: "Any" partition

absbxincycle12 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(68,1),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# OR of the two filter above using the VPSet mechanism

absbxincycle13 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(68,1),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(5,7),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# with one negative boundary: "Any" partition

absbxincycle14 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(-1,3),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# another with one negative boundary: "Any" partition

absbxincycle15 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(65,-1),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# OR with negative boundaries: "Any" partition

absbxincycle16 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(65,-1),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(20,20),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(-1,3),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )
# a few combined filters between Latency Aware and not Latency Aware cuts

absbxincycle21 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(65,-1),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(20,20),
                            partitionName = cms.untracked.string("Any")
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(5,8),
                            partitionName = cms.untracked.string("Any")
                         )
                       )
                     )

# partition name in the common part

absbxincycle22 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases"),
                           partitionName = cms.untracked.string("Any")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(65,-1),
                         ),
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(20,20),
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(5,8),
                         )
                       )
                     )

# a combined filter with different partitions

absbxincycle31 = cms.EDFilter("EventWithHistoryEDFilter",
                       commonConfiguration = cms.untracked.PSet(
                           historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                           APVPhaseLabel = cms.untracked.string("APVPhases")
                       ),
                       filterConfigurations = cms.untracked.VPSet(
                         cms.PSet(
                            absBXInCycleRange = cms.untracked.vint32(20,20),
                            partitionName = cms.untracked.string("TO")
                         ),
                         cms.PSet(
                            absBXInCycleRangeLtcyAware = cms.untracked.vint32(5,8),
                            partitionName = cms.untracked.string("TM")
                         )
                       )
                     )


import DPGAnalysis.SiStripTools.eventtimedistribution_cfi

etdabsbxincycle1 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle2 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle3 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle4 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle5 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle6 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()

etdabsbxincycle11 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle12 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle13 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle14 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle15 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle16 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()

etdabsbxincycle21 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()
etdabsbxincycle22 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()

etdabsbxincycle31 = DPGAnalysis.SiStripTools.eventtimedistribution_cfi.eventtimedistribution.clone()

absbxincycles1 = cms.Sequence(absbxincycle1 + etdabsbxincycle1)
absbxincycles2 = cms.Sequence(absbxincycle2 + etdabsbxincycle2)
absbxincycles3 = cms.Sequence(absbxincycle3 + etdabsbxincycle3)
absbxincycles4 = cms.Sequence(absbxincycle4 + etdabsbxincycle4)
absbxincycles5 = cms.Sequence(absbxincycle5 + etdabsbxincycle5)
absbxincycles6 = cms.Sequence(absbxincycle6 + etdabsbxincycle6)

absbxincycles11 = cms.Sequence(absbxincycle11 + etdabsbxincycle11)
absbxincycles12 = cms.Sequence(absbxincycle12 + etdabsbxincycle12)
absbxincycles13 = cms.Sequence(absbxincycle13 + etdabsbxincycle13)
absbxincycles14 = cms.Sequence(absbxincycle14 + etdabsbxincycle14)
absbxincycles15 = cms.Sequence(absbxincycle15 + etdabsbxincycle15)
absbxincycles16 = cms.Sequence(absbxincycle16 + etdabsbxincycle16)

absbxincycles21 = cms.Sequence(absbxincycle21 + etdabsbxincycle21)
absbxincycles22 = cms.Sequence(absbxincycle22 + etdabsbxincycle22)

absbxincycles31 = cms.Sequence(absbxincycle31 + etdabsbxincycle31)
