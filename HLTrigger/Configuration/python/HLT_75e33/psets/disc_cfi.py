import FWCore.ParameterSet.Config as cms

disc = cms.PSet(
    denominator = cms.VInputTag(
        cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probXqq"), cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probQCDbb"), cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probQCDcc"), cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probQCDb"), cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probQCDc"),
        cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probQCDothers")
    ),
    name = cms.string('XqqvsQCD'),
    numerator = cms.VInputTag(cms.InputTag("pfMassDecorrelatedParticleNetJetTags","probXqq"))
)