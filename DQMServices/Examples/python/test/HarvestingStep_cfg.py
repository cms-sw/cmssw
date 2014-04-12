import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("DQMServices.Examples.test.HarvestingAnalyzer_cfi")

process.load("DQMServices.Examples.test.HarvestingDataCertification_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:MEtoEDMConverter.root')
)

process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQMServices/Examples/test/QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1)
)

process.p1 = cms.Path(process.EDMtoMEConverter*process.harvestinganalyzer*process.qTester*process.harvestingdatacertification*process.dqmSaver)
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/ConverterTester/Test/RECO'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1

