import FWCore.ParameterSet.Config as cms

consecutiveHEs = cms.EDProducer('EventWithHistoryProducer',
                               historyDepth = cms.untracked.uint32(5)
                               )
