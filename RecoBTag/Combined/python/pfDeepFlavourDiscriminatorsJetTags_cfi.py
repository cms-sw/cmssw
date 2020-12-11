import FWCore.ParameterSet.Config as cms

pfDeepFlavourDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('BvsAll'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepFlavourJetTags', 'probb'),
            cms.InputTag('pfDeepFlavourJetTags', 'probbb'),
            cms.InputTag('pfDeepFlavourJetTags', 'problepb'),
            ),
         denominator = cms.VInputTag(),
         ),
      cms.PSet(
         name = cms.string('CvsB'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepFlavourJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepFlavourJetTags', 'probc'),
            cms.InputTag('pfDeepFlavourJetTags', 'probb'),
            cms.InputTag('pfDeepFlavourJetTags', 'probbb'),
            cms.InputTag('pfDeepFlavourJetTags', 'problepb'),
            ),
         ),
      cms.PSet(
         name = cms.string('CvsL'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepFlavourJetTags', 'probc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepFlavourJetTags', 'probc'),
            cms.InputTag('pfDeepFlavourJetTags', 'probuds'),
            cms.InputTag('pfDeepFlavourJetTags', 'probg'),
            ),
         ),
      )
   )
