import FWCore.ParameterSet.Config as cms

hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminator = cms.EDProducer( "PFTauDiscriminatorLogicalAndProducer",
    Prediscriminants = cms.PSet( 
      BooleanOperator = cms.string( "or" ),
      discr1 = cms.PSet( 
        cut = cms.double( 0.5 ),
        Producer = cms.InputTag( "hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator" )
      ),
      discr2 = cms.PSet( 
        cut = cms.double( 0.5 ),
        Producer = cms.InputTag( "hltHpsPFTauMediumRelativeChargedIsolationDiscriminator" )
      )
    ),
    PassValue = cms.double( 1.0 ),
    FailValue = cms.double( 0.0 ),
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" )
)
