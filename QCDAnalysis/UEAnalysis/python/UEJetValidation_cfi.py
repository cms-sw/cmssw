import FWCore.ParameterSet.Config as cms

UEJetValidationParameters = cms.PSet(
    RecoCaloJetCollectionName = cms.untracked.InputTag("iterativeCone5CaloJets"),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    etaLimit = cms.double(2.),
    selectedHLTBits = cms.vstring( 'HLT_MinBiasPixel',
                                   'HLT_MinBiasHcal',
                                   'HLT_MinBiasEcal',
                                   'HLT_MinBias',
                                   'HLT_ZeroBias',
                                   'HLT_Jet30',
                                   'HLT_Jet50',
                                   'HLT_Jet80',
                                   'HLT_Jet110',
                                   'HLT_Jet180',
                                   'HLT_Jet250' )
    )

UEJetValidation900 = cms.EDProducer("UEJetValidation",
                                    UEJetValidationParameters,
                                    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet"),
                                    TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet"),
                                    pTThreshold = cms.double(0.9)
)

UEJetValidation500 = cms.EDProducer("UEJetValidation",
                                    UEJetValidationParameters,
                                    ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet500"),
                                    TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet500"),
                                    pTThreshold = cms.double(0.5)
                                    )

UEJetValidation1500 = cms.EDProducer("UEJetValidation",
                                     UEJetValidationParameters,
                                     ChgGenJetCollectionName = cms.untracked.InputTag("IC5ChgGenJet1500"),
                                     TracksJetCollectionName = cms.untracked.InputTag("IC5TracksJet1500"),
                                     pTThreshold = cms.double(1.5)
                                     )

UEJetValidation = cms.Sequence(UEJetValidation900*UEJetValidation500*UEJetValidation1500)



