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

JetMultiplicity = cms.EDProducer("UEJetMultiplicity",
                                         UEJetMultiplicityParameters,
                                         genEventScale           = cms.InputTag("genEventScale"),
                                         eventScaleMin           = cms.double(  0. ), 
                                         eventScaleMax           = cms.double( 14000. )
                                         )

UEJetMultiplicitySummer08QCDDijet = cms.Sequence(JetMultiplicity)


