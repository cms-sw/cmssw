import FWCore.ParameterSet.Config as cms

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    jobReportOutputOnly = cms.untracked.bool(True)
)
