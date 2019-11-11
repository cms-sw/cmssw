import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

################# Quality Tests for jets #########################
from DQMServices.Core.DQMQualityTester import DQMQualityTester
qTesterJet = DQMQualityTester(
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/JetQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(False)
 )

################# Quality Tests for MET #########################
qTesterMET = DQMQualityTester(
     qtList = cms.untracked.FileInPath('DQMOffline/JetMET/test/METQualityTests.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(False)
 )

################# Data Certification #########################
dataCertificationJetMET = DQMEDHarvester('DataCertificationJetMET',
                              fileName       = cms.untracked.string(""),
                              refFileName    = cms.untracked.string(""),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMDataCertificationResult.root"),
                              Verbose        = cms.untracked.int32(0),
                              metFolder      = cms.untracked.string("Cleaned"),
                              jetAlgo        = cms.untracked.string("ak4"),
                              folderName     = cms.untracked.string("JetMET/EventInfo"),  
                              METTypeRECO    = cms.InputTag("pfMETT1"),
                              #for the uncleaned directory the flag needs to be set accordingly in
                              #metDQMConfig_cfi.py
                              METTypeRECOUncleaned = cms.InputTag("pfMet"),
                              METTypeMiniAOD = cms.InputTag("slimmedMETs"),
                              JetTypeRECO    = cms.InputTag("ak4PFJetsCHS"),
                              JetTypeMiniAOD = cms.InputTag("slimmedJets"),
                              #if changed here, change METAnalyzer module in same manner and jetDQMconfig
                              etaBin      = cms.int32(100),
                              etaMax      = cms.double(5.0),
                              etaMin      = cms.double(-5.0),
                              pVBin       = cms.int32(100),
                              pVMax       = cms.double(100.0),
                              pVMin       = cms.double(0.0),
                              ptBin       = cms.int32(100),
                              ptMax       = cms.double(500.0),
                              ptMin       = cms.double(20.0),
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

                              isHI                    = cms.untracked.bool(False),
)

dataCertificationJetMETHI = dataCertificationJetMET.clone(
    isHI    = cms.untracked.bool(True),
    jetAlgo = cms.untracked.string("ak"),
)
