import FWCore.ParameterSet.Config as cms


DaqData = cms.EDFilter("DQMDaqInfo",
    saveDCFile = cms.untracked.bool(True),
    outputFile = cms.string('dqmDaqInfo.txt'),
    saveRootFile = cms.bool(True),
    FedGranularityInput = cms.untracked.bool(False),
    OutputFileName = cms.string('DQMDAQDataCert.root')
)
