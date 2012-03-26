import FWCore.ParameterSet.Config as cms

eventtimedistribution = cms.EDAnalyzer('EventTimeDistribution',
                                       historyProduct = cms.InputTag("consecutiveHEs"),
                                       apvPhaseCollection = cms.InputTag("APVPhases"),
                                       phasePartition = cms.untracked.string("All"),
                                       maxLSBeforeRebin = cms.untracked.uint32(100),
                                       wantDBXvsBXincycle = cms.untracked.bool(True),
                                       wantDBXvsBX = cms.untracked.bool(False),
                                       wantBXincyclevsBX = cms.untracked.bool(False),
                                       wantOrbitvsBXincycle = cms.untracked.bool(False)
)	
