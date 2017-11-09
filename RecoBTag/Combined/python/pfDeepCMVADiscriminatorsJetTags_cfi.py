import FWCore.ParameterSet.Config as cms

pfDeepCMVADiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCMVAJetTags', 'probb'),
            cms.InputTag('pfDeepCMVAJetTags', 'probbb'),
            ),
         denominator = cms.VInputTag(),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCMVAJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepCMVAJetTags', 'probc'),
            cms.InputTag('pfDeepCMVAJetTags', 'probb'),
            cms.InputTag('pfDeepCMVAJetTags', 'probbb'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCMVAJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepCMVAJetTags', 'probc'),
            cms.InputTag('pfDeepCMVAJetTags', 'probudsg'),
            ),
         ),
      )
   )
