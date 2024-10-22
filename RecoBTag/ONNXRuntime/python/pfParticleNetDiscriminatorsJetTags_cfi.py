import FWCore.ParameterSet.Config as cms

pfParticleNetDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('TvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probTbcq'),
            cms.InputTag('pfParticleNetJetTags', 'probTbqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probTbcq'),
            cms.InputTag('pfParticleNetJetTags', 'probTbqq'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('WvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probWcq'),
            cms.InputTag('pfParticleNetJetTags', 'probWqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probWcq'),
            cms.InputTag('pfParticleNetJetTags', 'probWqq'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ZvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZbb'),
            cms.InputTag('pfParticleNetJetTags', 'probZcc'),
            cms.InputTag('pfParticleNetJetTags', 'probZqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZbb'),
            cms.InputTag('pfParticleNetJetTags', 'probZcc'),
            cms.InputTag('pfParticleNetJetTags', 'probZqq'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ZbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ZccvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probZcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('HbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('HccvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('H4qvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHqqqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetJetTags', 'probHqqqq'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDbb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDcc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDb'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDc'),
            cms.InputTag('pfParticleNetJetTags', 'probQCDothers'),
            ),
         ),
      )
   )
