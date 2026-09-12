import FWCore.ParameterSet.Config as cms

hltParticleTransformerDiscriminatorsJetTags = cms.EDProducer("BTagProbabilityToDiscriminator",
    discriminators = cms.VPSet(
      cms.PSet(name = cms.string("TauvsAll"),
        numerator = cms.VInputTag('hltParticleTransformerONNXJetTags:probtaup',
        'hltParticleTransformerONNXJetTags:probtaum'),
        denominator = cms.VInputTag(
            'hltParticleTransformerONNXJetTags:probb',
            'hltParticleTransformerONNXJetTags:probg',
            'hltParticleTransformerONNXJetTags:probtaup',
            'hltParticleTransformerONNXJetTags:probtaum',
            'hltParticleTransformerONNXJetTags:probc',
            'hltParticleTransformerONNXJetTags:probuds'
        )
      )
    )
)
