import FWCore.ParameterSet.Config as cms

pfParticleTransformerAK4DiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'problepb'),
            ),
         denominator=cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'problepb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probc'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probg'),
         ),
      ),
      cms.PSet(
         name = cms.string('BvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'problepb'),
            ),
         denominator=cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'problepb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probg'),
         ),
      ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probc'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probc'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'problepb'),
            ),
         ),
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probuds'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleTransformerAK4JetTags', 'probg'),
            ),
         ),

      )
   )
