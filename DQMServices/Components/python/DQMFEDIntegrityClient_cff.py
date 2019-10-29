import FWCore.ParameterSet.Config as cms

dqmFEDIntegrity = cms.EDAnalyzer("DQMFEDIntegrityClient",
   fillInEventloop = cms.untracked.bool(True),
   fillOnEndLumi = cms.untracked.bool(True),
   fillOnEndRun = cms.untracked.bool(True),
   fillOnEndJob = cms.untracked.bool(False),
   moduleName = cms.untracked.string('FED'),
   fedFolderName = cms.untracked.string('FEDIntegrity')
   )

from DQMServices.Core.DQMQualityTester import DQMQualityTester
dqmQTestDQMFED = DQMQualityTester(
   prescaleFactor = cms.untracked.int32(1),
   qtList = cms.untracked.FileInPath('DQMServices/Components/data/DQMFEDQualityTests.xml'),
   getQualityTestsFromFile = cms.untracked.bool(True),
   qtestOnEndLumi = cms.untracked.bool(True),
   qtestOnEndRun = cms.untracked.bool(True),
   qtestOnEndJob = cms.untracked.bool(False)
   )

dqmFEDIntegrityClient = cms.Sequence(dqmFEDIntegrity*dqmQTestDQMFED)
