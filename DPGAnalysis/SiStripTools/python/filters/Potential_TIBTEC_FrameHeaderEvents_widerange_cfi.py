import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsWide = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TI")
                                                ),
                                             filterConfigurations = cms.untracked.VPSet(
                                               cms.PSet(
                                                 absBXInCycleRangeLtcyAware = cms.untracked.vint32(16,44)
                                               )
                                             )
                                           )
# foo bar baz
# UGUJng4CKHIzJ
# ujCHKymhUtW8B
