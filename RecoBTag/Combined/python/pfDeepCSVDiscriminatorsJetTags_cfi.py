import FWCore.ParameterSet.Config as cms

pfDeepCSVDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCSVJetTags', 'probb'),
            cms.InputTag('pfDeepCSVJetTags', 'probbb'),
            ),
         denominator = cms.VInputTag(),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCSVJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepCSVJetTags', 'probc'),
            cms.InputTag('pfDeepCSVJetTags', 'probb'),
            cms.InputTag('pfDeepCSVJetTags', 'probbb'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepCSVJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepCSVJetTags', 'probudsg'),
            cms.InputTag('pfDeepCSVJetTags', 'probc'),
            ),
         ),
      )
   )
