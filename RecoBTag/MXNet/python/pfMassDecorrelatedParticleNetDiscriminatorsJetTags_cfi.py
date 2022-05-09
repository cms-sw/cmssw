import FWCore.ParameterSet.Config as cms

pfMassDecorrelatedParticleNetDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('XbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXbb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('XccvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXcc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('XqqvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probXqq'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedParticleNetJetTags', 'probQCDothers'),
            ),
         ),

      )
   )
