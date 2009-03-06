import FWCore.ParameterSet.Config as cms

UEJetValidationParameters = cms.PSet(
    CaloJetCollectionName   = cms.untracked.InputTag("sisCone5CaloJets"),
    ChgGenJetCollectionName = cms.untracked.InputTag("ueSisCone5ChgGenJet"),
    TracksJetCollectionName = cms.untracked.InputTag("ueSisCone5TracksJet"),
    triggerEvent            = cms.InputTag("hltTriggerSummaryAOD"),
    triggerResults          = cms.InputTag("TriggerResults","","HLT"),
    pTThreshold             = cms.double( 0.9 ),
    etaLimit                = cms.double( 2.  ),
    dRLimitForMatching      = cms.double( 0.5 ),
    selectedHLTBits = cms.vstring( 'HLTMinBiasPixel',
                                   'HLTMinBiasHcal',
                                   'HLTMinBiasEcal',
                                   'HLTMinBias',
                                   'HLTZeroBias',
                                   'HLT1jet30',
                                   'HLT1jet50',
                                   'HLT1jet80',
                                   'HLT1jet110',
                                   'HLT1jet180',
                                   'HLT1jet250' )
    )

CSA08_0_pT_30_threshold900 = cms.EDProducer("UEJetValidation",
                                            UEJetValidationParameters,
                                            eventScaleMin           = cms.double(  0. ), 
                                            eventScaleMax           = cms.double( 30. )
                                            )
CSA08_30_pT_45_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double( 30. ), 
                                             eventScaleMax           = cms.double( 45. )
                                             )
CSA08_45_pT_75_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double( 45. ), 
                                             eventScaleMax           = cms.double( 75. )
                                             )
CSA08_75_pT_120_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double(  75. ), 
                                             eventScaleMax           = cms.double( 120. )
                                             )
CSA08_120_pT_160_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double( 120. ), 
                                             eventScaleMax           = cms.double( 160. )
                                             )
CSA08_160_pT_220_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double( 160. ), 
                                             eventScaleMax           = cms.double( 220. )
                                             )
CSA08_220_pT_threshold900 = cms.EDProducer("UEJetValidation",
                                             UEJetValidationParameters,
                                             eventScaleMin           = cms.double( 220.  ), 
                                             eventScaleMax           = cms.double( 7000. )
                                             )


#CSA08_0_pT_30_threshold900_Corrected = cms.EDProducer("UEJetValidation",
#                                                      UEJetValidationParameters,
#                                                      eventScaleMin           = cms.double(  0. ), 
#                                                      eventScaleMax           = cms.double( 30. ) 
#                                                      )
#CSA08_0_pT_30_threshold900_Corrected.CaloJetCollectionName = 'L2L3CorJetScone5'

#UEJetValidation = cms.Sequence(CSA08_0_pT_30_threshold900*CSA08_0_pT_30_threshold900_Corrected)
UEJetValidation = cms.Sequence(CSA08_0_pT_30_threshold900*CSA08_30_pT_45_threshold900*CSA08_45_pT_75_threshold900*CSA08_75_pT_120_threshold900*CSA08_120_pT_160_threshold900*CSA08_160_pT_220_threshold900*CSA08_220_pT_threshold900)



