import FWCore.ParameterSet.Config as cms

pfParticleNetAK4DiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probb'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probbb'),
            ),
         denominator=cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probb'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probbb'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probcc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probg'),
         ),
      ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probcc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probg'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probcc'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probb'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probbb'),
            ),
         ),
      cms.PSet(
         name = cms.string('QvsG'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probuds'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetAK4JetTags', 'probuds'),
            cms.InputTag('pfParticleNetAK4JetTags', 'probg'),
            ),
         ),

      )
   )
