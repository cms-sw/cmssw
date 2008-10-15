import FWCore.ParameterSet.Config as cms

UEJetValidationParameters = cms.PSet(
    CaloJetCollectionName = cms.untracked.InputTag("sisCone5CaloJets"),
    triggerEvent   = cms.InputTag("hltTriggerSummaryAOD"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    eventScaleMin           = cms.double( 0.    ), 
    eventScaleMax           = cms.double( 7000. ), 
    etaLimit                = cms.double( 2.    ),
    dRByPiLimitForMatching  = cms.double( 0.1   ),
    pTratioRangeForMatching = cms.double( 0.1   ),
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
                                    ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet"),
                                    TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet"),
                                    pTThreshold = cms.double(0.9)
                                    )
UECorrectedJetValidation900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet"),
                                             TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet"),
                                             pTThreshold = cms.double(0.9)
                                             )
UECorrectedJetValidation900.CaloJetCollectionName = 'L2L3CorJetScone5'

UEJetValidation500 = cms.EDProducer("UEJetValidation",
                                    UEJetValidationParameters,
                                    ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet500"),
                                    TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet500"),
                                    pTThreshold = cms.double(0.5)
                                    )
UECorrectedJetValidation500 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet500"),
                                             TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet500"),
                                             pTThreshold = cms.double(0.5)
                                             )
UECorrectedJetValidation500.CaloJetCollectionName = 'L2L3CorJetScone5'

UEJetValidation1500 = cms.EDProducer("UEJetValidation",
                                     UEJetValidationParameters,
                                     ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet1500"),
                                     TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet1500"),
                                     pTThreshold = cms.double(1.5)
                                     )
UECorrectedJetValidation1500 = cms.EDProducer("UEJetValidation",
                                              UEJetValidationParameters,
                                              ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet1500"),
                                              TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet1500"),
                                              pTThreshold = cms.double(1.5)
                                              )
UECorrectedJetValidation1500.CaloJetCollectionName = 'L2L3CorJetScone5'

UEJetValidation = cms.Sequence(UEJetValidation900*UECorrectedJetValidation900*UEJetValidation500*UECorrectedJetValidation500*UEJetValidation1500*UECorrectedJetValidation1500)



