import FWCore.ParameterSet.Config as cms

UEAnalysisRootple = cms.EDProducer("AnalysisRootpleProducer",
                                   TracksCollectionName      = cms.InputTag("goodTracks"),
                                   RecoCaloJetCollectionName = cms.InputTag("iterativeCone5CaloJets"),
                                   ChgGenJetCollectionName   = cms.InputTag("IC5ChgGenJet"),
                                   MCEvent                   = cms.InputTag("generator"),
                                   TracksJetCollectionName   = cms.InputTag("IC5TracksJet"),
                                   triggerEvent              = cms.InputTag("hltTriggerSummaryAOD"),
                                   ChgGenPartCollectionName  = cms.InputTag("chargeParticles"),
                                   OnlyRECO                  = cms.bool(True),
                                   GenJetCollectionName      = cms.InputTag("IC5GenJet"),
                                   triggerResults            = cms.InputTag("TriggerResults","","HLT"),
                                   genEventScale             = cms.InputTag("generator") 
                                   )
#/// Pythia: genEventScale = cms.InputTag("genEventScale")
#/// Herwig: genEventScale = cms.InputTag("genEventKTValue")

UEAnalysisRootple500 = cms.EDProducer("AnalysisRootpleProducer",
                                      TracksCollectionName      = cms.InputTag("goodTracks"),
                                      RecoCaloJetCollectionName = cms.InputTag("iterativeCone5CaloJets"),
                                      ChgGenJetCollectionName   = cms.InputTag("IC5ChgGenJet500"),
                                      MCEvent                   = cms.InputTag("generator"),
                                      TracksJetCollectionName   = cms.InputTag("IC5TracksJet500"),
                                      triggerEvent              = cms.InputTag("hltTriggerSummaryAOD"),
                                      ChgGenPartCollectionName  = cms.InputTag("chargeParticles"),
                                      OnlyRECO                  = cms.bool(True),
                                      GenJetCollectionName      = cms.InputTag("IC5GenJet500"),
                                      triggerResults            = cms.InputTag("TriggerResults","","HLT"),
                                      genEventScale             = cms.InputTag("generator") 
)

UEAnalysisRootple1500 = cms.EDProducer("AnalysisRootpleProducer",
                                       TracksCollectionName      = cms.InputTag("goodTracks"),
                                       RecoCaloJetCollectionName = cms.InputTag("iterativeCone5CaloJets"),
                                       ChgGenJetCollectionName   = cms.InputTag("IC5ChgGenJet1500"),
                                       MCEvent                   = cms.InputTag("generator"),
                                       TracksJetCollectionName   = cms.InputTag("IC5TracksJet1500"),
                                       triggerEvent              = cms.InputTag("hltTriggerSummaryAOD"),
                                       ChgGenPartCollectionName  = cms.InputTag("chargeParticles"),
                                       OnlyRECO                  = cms.bool(True),
                                       GenJetCollectionName      = cms.InputTag("IC5GenJet1500"),
                                       triggerResults            = cms.InputTag("TriggerResults","","HLT"),
                                       genEventScale             = cms.InputTag("generator")
)

UEAnalysis = cms.Sequence(UEAnalysisRootple*UEAnalysisRootple500*UEAnalysisRootple1500)



