import FWCore.ParameterSet.Config as cms

dtEfficiencyMonitor = cms.EDFilter("DTEfficiencyTask",
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits')
)


