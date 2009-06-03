import FWCore.ParameterSet.Config as cms

dqmFEDIntegrity = cms.EDFilter("DQMFEDIntegrityClient",
   fillOnEndRun = cms.untracked.bool(False),
   fillOnEndLumi = cms.untracked.bool(True),
   fillOnEndJob = cms.untracked.bool(False),
   fillInEventloop = cms.untracked.bool(True),
   moduleName = cms.untracked.string('FED')
   )

dqmQTestDQMFED = cms.EDFilter("QualityTester",
   prescaleFactor = cms.untracked.int32(1),
   qtList = cms.untracked.FileInPath('DQMServices/Components/data/DQMFEDQualityTests.xml'),
   getQualityTestsFromFile = cms.untracked.bool(True),
   qtestOnEndRun = cms.untracked.bool(True)
   )

dqmFEDIntegrityClient = cms.Sequence(dqmFEDIntegrity*dqmQTestDQMFED)
