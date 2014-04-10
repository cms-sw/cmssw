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
                              metFolder      = cms.untracked.string("Cleaned"),
                              jetAlgo        = cms.untracked.string("ak5"),
                              folderName     = cms.untracked.string("JetMET/EventInfo"),  
                              pfBarrelJetMeanTest   = cms.untracked.bool(True),
                              pfBarrelJetKSTest     = cms.untracked.bool(False),
                              pfEndcapJetMeanTest   = cms.untracked.bool(True),
                              pfEndcapJetKSTest     = cms.untracked.bool(False),
                              pfForwardJetMeanTest  = cms.untracked.bool(True),
                              pfForwardJetKSTest    = cms.untracked.bool(False),
                              caloJetMeanTest           = cms.untracked.bool(True),
                              caloJetKSTest             = cms.untracked.bool(False),
                              jptJetMeanTest          = cms.untracked.bool(False),
                              jptJetKSTest            = cms.untracked.bool(False),                                        
                              caloMETMeanTest         = cms.untracked.bool(True),
                              caloMETKSTest           = cms.untracked.bool(False),
                              pfMETMeanTest           = cms.untracked.bool(True),
                              pfMETKSTest             = cms.untracked.bool(False),
                              tcMETMeanTest           = cms.untracked.bool(False),
                              tcMETKSTest             = cms.untracked.bool(False),

)


