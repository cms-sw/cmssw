import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfParticleNetTagInfos_cfi import pfParticleNetTagInfos
from RecoBTag.MXNet.pfParticleNetJetTags_cfi import pfParticleNetJetTags as _pfParticleNetJetTags
from RecoBTag.MXNet.Parameters.ParticleNet.V00.pfParticleNetPreprocessParams_cfi import pfParticleNetPreprocessParams
from RecoBTag.MXNet.Parameters.ParticleNet.V00.pfMassDecorrelatedParticleNetPreprocessParams_cfi import pfMassDecorrelatedParticleNetPreprocessParams
from RecoBTag.MXNet.pfParticleNetDiscriminatorsJetTags_cfi import pfParticleNetDiscriminatorsJetTags
from RecoBTag.MXNet.pfMassDecorrelatedParticleNetDiscriminatorsJetTags_cfi import pfMassDecorrelatedParticleNetDiscriminatorsJetTags

pfParticleNetJetTags = _pfParticleNetJetTags.clone(
    preprocessParams = pfParticleNetPreprocessParams,
    model_path = 'RecoBTag/Combined/data/ParticleNet/V00/ParticleNet-symbol.json',
    param_path = 'RecoBTag/Combined/data/ParticleNet/V00/ParticleNet-0000.params',
    debugMode  = False, # debug
)

pfMassDecorrelatedParticleNetJetTags = _pfParticleNetJetTags.clone(
    preprocessParams = pfMassDecorrelatedParticleNetPreprocessParams,
    model_path = 'RecoBTag/Combined/data/ParticleNet-MD/V00/ParticleNet-symbol.json',
    param_path = 'RecoBTag/Combined/data/ParticleNet-MD/V00/ParticleNet-0000.params',
    flav_names = cms.vstring("probXbb", "probXcc", "probXqq", "probQCDbb", "probQCDcc", "probQCDb", "probQCDc", "probQCDothers"),
    debugMode = False,  # debug
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfParticleNetTask = cms.Task(puppi, primaryVertexAssociation, pfParticleNetTagInfos,
                             pfParticleNetJetTags, pfMassDecorrelatedParticleNetJetTags, pfParticleNetDiscriminatorsJetTags)

# declare all the discriminators
# nominal: probs
_pfParticleNetJetTagsProbs = ['pfParticleNetJetTags:' + flav_name
                              for flav_name in pfParticleNetJetTags.flav_names]
# nominal: meta-taggers
_pfParticleNetJetTagsMetaDiscrs = ['pfParticleNetDiscriminatorsJetTags:' + disc.name.value()
                                   for disc in pfParticleNetDiscriminatorsJetTags.discriminators]
# mass-decorrelated: probs
_pfMassDecorrelatedParticleNetJetTagsProbs = ['pfMassDecorrelatedParticleNetJetTags:' + flav_name
                              for flav_name in pfMassDecorrelatedParticleNetJetTags.flav_names]
# mass-decorrelated: meta-taggers
_pfMassDecorrelatedParticleNetJetTagsMetaDiscrs = ['pfMassDecorrelatedParticleNetDiscriminatorsJetTags:' + disc.name.value()
                                   for disc in pfMassDecorrelatedParticleNetDiscriminatorsJetTags.discriminators]

_pfParticleNetJetTagsAll = _pfParticleNetJetTagsProbs + _pfParticleNetJetTagsMetaDiscrs + \
    _pfMassDecorrelatedParticleNetJetTagsProbs + _pfMassDecorrelatedParticleNetJetTagsMetaDiscrs
