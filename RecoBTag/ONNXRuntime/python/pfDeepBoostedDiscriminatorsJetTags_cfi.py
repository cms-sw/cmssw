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
      cms.PSet(
         name = cms.string('ZvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probZqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probZqq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ZbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probZbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('HbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probHbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probHbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('H4qvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probHqqqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfDeepBoostedJetTags', 'probHqqqq'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      )
   )
