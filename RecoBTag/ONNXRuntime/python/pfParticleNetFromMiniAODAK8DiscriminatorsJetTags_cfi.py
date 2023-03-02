import FWCore.ParameterSet.Config as cms

pfParticleNetFromMiniAODAK8DiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('HbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHbb'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HccvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHcc'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HttvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHtt'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHtt'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HtmvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHtm'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHtm'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HtevsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHte'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHte'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HqqvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHqq'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      cms.PSet(
         name = cms.string('HggvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHgg'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probHgg'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD2hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD1hf'),
            cms.InputTag('pfParticleNetFromMiniAODAK8JetTags', 'probQCD0hf'),
            ),
         ),
      )
   )
