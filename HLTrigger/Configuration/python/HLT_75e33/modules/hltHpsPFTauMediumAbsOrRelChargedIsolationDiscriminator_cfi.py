import FWCore.ParameterSet.Config as cms

hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminator = cms.EDProducer("PFTauDiscriminatorLogicalAndProducer",
    FailValue = cms.double(0.0),
    PFTauProducer = cms.InputTag("hltHpsPFTauProducer"),
    PassValue = cms.double(1.0),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('or'),
        discr1 = cms.PSet(
            Producer = cms.InputTag("hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator"),
            cut = cms.double(0.5)
        ),
        discr2 = cms.PSet(
            Producer = cms.InputTag("hltHpsPFTauMediumRelativeChargedIsolationDiscriminator"),
            cut = cms.double(0.5)
        )
    )
)
