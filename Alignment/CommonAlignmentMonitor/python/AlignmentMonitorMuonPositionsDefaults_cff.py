import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonPositionsDefaults = cms.PSet(
    collectorActive = cms.bool(False),
    collectorPath = cms.string('./'),
    outpath = cms.string('./'),
    collectorNJobs = cms.int32(0)
)

