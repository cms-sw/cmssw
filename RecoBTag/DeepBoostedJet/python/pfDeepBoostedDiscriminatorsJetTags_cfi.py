import FWCore.ParameterSet.Config as cms

pfDeepBoostedDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('TvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probTbcq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probTbqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probTbcq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probTbqq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('WvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probWcq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probWqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probWcq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probWqq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      )
   )
