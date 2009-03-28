import FWCore.ParameterSet.Config as cms

################# Quality Tests #########################
qTesterJetMET = cms.EDFilter("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/JetMETQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(False)
 )

################# Data Certification #########################
dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string(""),
                              refFileName    = cms.untracked.string(""),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMDataCertificationResult.root"),
                              Verbose        = cms.untracked.int32(0),
                              TestType       = cms.untracked.int32(0)
)
