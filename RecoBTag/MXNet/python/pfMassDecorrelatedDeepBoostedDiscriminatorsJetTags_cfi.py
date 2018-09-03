import FWCore.ParameterSet.Config as cms

pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags = cms.EDProducer(
   'BTagProbabilityToDiscriminator',
   discriminators = cms.VPSet(
      cms.PSet(
         name = cms.string('TvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probTbcq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probTbqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probTbcq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probTbqq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('WvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probWcq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probWqq'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probWcq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probWqq'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),

      cms.PSet(
         name = cms.string('ZHbbvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ZHccvsQCD'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),

      cms.PSet(
         name = cms.string('bbvsLight'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),
      cms.PSet(
         name = cms.string('ccvsLight'),
         numerator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            ),
         denominator = cms.VInputTag(
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probZcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probHcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDbb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDcc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDb'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDc'),
            cms.InputTag('pfMassDecorrelatedDeepBoostedJetTags', 'probQCDothers'),
            ),
         ),

      )
   )
