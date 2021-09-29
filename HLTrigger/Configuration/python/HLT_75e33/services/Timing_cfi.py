import FWCore.ParameterSet.Config as cms

Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True)
)
