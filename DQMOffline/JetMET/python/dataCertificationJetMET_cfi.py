import FWCore.ParameterSet.Config as cms

################# Quality Tests for jets #########################
qTesterJet = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/JetQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(True)
 )

################# Quality Tests for MET #########################
qTesterMET = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/METQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(True)
 )

################# Data Certification #########################
dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string(""),
                              refFileName    = cms.untracked.string(""),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMDataCertificationResult.root"),
                              Verbose        = cms.untracked.int32(1),
                              TestType       = cms.untracked.int32(0),
                              jet_ks_thresh  = cms.untracked.double(0.000001),
                              met_ks_thresh  = cms.untracked.double(0.00000001),
                              met_phi_thresh = cms.untracked.double(0.75)
)


