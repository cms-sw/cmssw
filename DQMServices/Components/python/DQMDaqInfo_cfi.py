import FWCore.ParameterSet.Config as cms


dqmDaqInfo = cms.EDFilter("DQMDaqInfo",
    saveDCFile = cms.untracked.bool(False),
    outputFile = cms.string('dqmDaqInfo.txt'),
)
