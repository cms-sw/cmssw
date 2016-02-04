import FWCore.ParameterSet.Config as cms

################# Quality Tests for jets #########################
qTesterJet = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/JetQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(False)
 )

################# Quality Tests for MET #########################
qTesterMET = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/METQualityTests.xml'),
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
                              metFolder      = cms.untracked.string("LowPtJet"),
                              caloBarrelJetMeanTest   = cms.untracked.bool(True),
                              caloBarrelJetKSTest     = cms.untracked.bool(False),
                              caloEndcapJetMeanTest   = cms.untracked.bool(True),
                              caloEndcapJetKSTest     = cms.untracked.bool(False),
                              caloForwardJetMeanTest  = cms.untracked.bool(True),
                              caloForwardJetKSTest    = cms.untracked.bool(False),
                              pfJetMeanTest           = cms.untracked.bool(True),
                              pfJetKSTest             = cms.untracked.bool(False),
                              jptJetMeanTest          = cms.untracked.bool(True),
                              jptJetKSTest            = cms.untracked.bool(False),
                                         
                              caloMETMeanTest         = cms.untracked.bool(True),
                              caloMETKSTest           = cms.untracked.bool(False),
                              calonohfMETMeanTest     = cms.untracked.bool(True),
                              calonohfMETKSTest       = cms.untracked.bool(False),
                              pfMETMeanTest           = cms.untracked.bool(True),
                              pfMETKSTest             = cms.untracked.bool(False),
                              tcMETMeanTest           = cms.untracked.bool(True),
                              tcMETKSTest             = cms.untracked.bool(False),
                              muMETMeanTest           = cms.untracked.bool(True),
                              muMETKSTest             = cms.untracked.bool(False)


)


