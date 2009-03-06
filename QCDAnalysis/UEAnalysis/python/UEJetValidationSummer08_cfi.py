import FWCore.ParameterSet.Config as cms

UEJetValidationParameters = cms.PSet(
    #    CaloJetCollectionName   = cms.InputTag("sisCone5CaloJets"),
    CaloJetCollectionName   = cms.InputTag("L2L3CorJetSC5Calo"),
    ChgGenJetCollectionName = cms.InputTag("ueSisCone5ChgGenJet"),
    TracksJetCollectionName = cms.InputTag("ueSisCone5TracksJet"),
    triggerEvent            = cms.InputTag("hltTriggerSummaryAOD"),
    triggerResults          = cms.InputTag("TriggerResults","","HLT"),
    pTThreshold             = cms.double( 0.9 ),
    etaLimit                = cms.double( 2.  ),
    dRLimitForMatching      = cms.double( 0.5 ),
    selectedHLTBits         = cms.vstring( 'HLT_MinBiasPixel',
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

Summer08_0_pT_15_threshold900 = cms.EDProducer("UEJetValidation",
                                               UEJetValidationParameters,
                                               genEventScale           = cms.InputTag("genEventScale"),
                                               eventScaleMin           = cms.double(  0. ), 
                                               eventScaleMax           = cms.double( 15. )
                                               )
Summer08_15_pT_30_threshold900 = cms.EDProducer("UEJetValidation",
                                                UEJetValidationParameters,
                                                genEventScale           = cms.InputTag("genEventScale"),
                                                eventScaleMin           = cms.double( 15. ), 
                                                eventScaleMax           = cms.double( 30. )
                                                )
Summer08_30_pT_80_threshold900 = cms.EDProducer("UEJetValidation",
                                                UEJetValidationParameters,
                                                genEventScale           = cms.InputTag("genEventScale"),
                                                eventScaleMin           = cms.double( 30. ), 
                                                eventScaleMax           = cms.double( 80. )
                                                )
Summer08_80_pT_170_threshold900 = cms.EDProducer("UEJetValidation",
                                                 UEJetValidationParameters,
                                                 genEventScale           = cms.InputTag("genEventScale"),
                                                 eventScaleMin           = cms.double(  80. ), 
                                                 eventScaleMax           = cms.double( 170. )
                                                 )
Summer08_170_pT_300_threshold900 = cms.EDProducer("UEJetValidation",
                                                  UEJetValidationParameters,
                                                  genEventScale           = cms.InputTag("genEventScale"),
                                                  eventScaleMin           = cms.double( 170. ), 
                                                  eventScaleMax           = cms.double( 300. )
                                                  )
Summer08_300_pT_470_threshold900 = cms.EDProducer("UEJetValidation",
                                                  UEJetValidationParameters,
                                                  genEventScale           = cms.InputTag("genEventScale"),
                                                  eventScaleMin           = cms.double( 300. ), 
                                                  eventScaleMax           = cms.double( 470. )
                                                  )

UEJetValidation = cms.Sequence(Summer08_0_pT_15_threshold900*Summer08_15_pT_30_threshold900*Summer08_30_pT_80_threshold900*Summer08_80_pT_170_threshold900*Summer08_170_pT_300_threshold900*Summer08_300_pT_470_threshold900)


