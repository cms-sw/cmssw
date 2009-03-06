import FWCore.ParameterSet.Config as cms

UEJetMultiplicityParameters = cms.PSet(
    ChgGenJetCollectionName = cms.InputTag("ueKt4ChgGenJet"),
    TracksJetCollectionName = cms.InputTag("ueKt4TracksJet"),
    triggerEvent            = cms.InputTag("hltTriggerSummaryAOD"),
    triggerResults          = cms.InputTag("TriggerResults","","HLT"),
    pTThreshold             = cms.double( 0.9 ),
    etaLimit                = cms.double( 2.  ),
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

JetMultiplicity_0_pT_15 = cms.EDProducer("UEJetMultiplicity",
                                         UEJetMultiplicityParameters,
                                         genEventScale           = cms.InputTag("genEventScale"),
                                         eventScaleMin           = cms.double(  0. ), 
                                         eventScaleMax           = cms.double( 15. )
                                         )
JetMultiplicity_15_pT_30 = cms.EDProducer("UEJetMultiplicity",
                                          UEJetMultiplicityParameters,
                                          genEventScale           = cms.InputTag("genEventScale"),
                                          eventScaleMin           = cms.double( 15. ), 
                                          eventScaleMax           = cms.double( 30. )
                                          )
JetMultiplicity_30_pT_80 = cms.EDProducer("UEJetMultiplicity",
                                          UEJetMultiplicityParameters,
                                          genEventScale           = cms.InputTag("genEventScale"),
                                          eventScaleMin           = cms.double( 30. ), 
                                          eventScaleMax           = cms.double( 80. )
                                          )
JetMultiplicity_80_pT_170 = cms.EDProducer("UEJetMultiplicity",
                                           UEJetMultiplicityParameters,
                                           genEventScale           = cms.InputTag("genEventScale"),
                                           eventScaleMin           = cms.double(  80. ), 
                                           eventScaleMax           = cms.double( 170. )
                                           )
JetMultiplicity_170_pT_300 = cms.EDProducer("UEJetMultiplicity",
                                            UEJetMultiplicityParameters,
                                            genEventScale           = cms.InputTag("genEventScale"),
                                            eventScaleMin           = cms.double( 170. ), 
                                            eventScaleMax           = cms.double( 300. )
                                            )
JetMultiplicity_300_pT_470 = cms.EDProducer("UEJetMultiplicity",
                                            UEJetMultiplicityParameters,
                                            genEventScale           = cms.InputTag("genEventScale"),
                                            eventScaleMin           = cms.double( 300. ), 
                                            eventScaleMax           = cms.double( 470. )
                                            )

UEJetMultiplicitySummer08 = cms.Sequence(JetMultiplicity_0_pT_15*JetMultiplicity_15_pT_30*JetMultiplicity_30_pT_80*JetMultiplicity_80_pT_170*JetMultiplicity_170_pT_300*JetMultiplicity_300_pT_470)


