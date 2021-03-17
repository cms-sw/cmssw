import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.pfParticleNetDiscriminatorsJetTags_cfi import pfParticleNetDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfMassDecorrelatedParticleNetDiscriminatorsJetTags_cfi import pfMassDecorrelatedParticleNetDiscriminatorsJetTags

pfParticleNetTagInfos = pfDeepBoostedJetTagInfos.clone(
    use_puppiP4 = False
)

pfParticleNetJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/particle-net.onnx',
    flav_names = ["probTbcq",  "probTbqq",  "probTbc",   "probTbq",  "probTbel", "probTbmu", "probTbta",
                  "probWcq",   "probWqq",   "probZbb",   "probZcc",  "probZqq",  "probHbb", "probHcc",
                  "probHqqqq", "probQCDbb", "probQCDcc", "probQCDb", "probQCDc", "probQCDothers"],
)

pfMassDecorrelatedParticleNetJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/particle-net.onnx',
    flav_names = ["probXbb", "probXcc", "probXqq", "probQCDbb", "probQCDcc",
                  "probQCDb", "probQCDc", "probQCDothers"],
)

pfParticleNetMassRegressionJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/particle-net.onnx',
    flav_names = ["mass"],
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfParticleNetTask = cms.Task(puppi, primaryVertexAssociation, pfParticleNetTagInfos,
                             pfParticleNetJetTags, pfMassDecorrelatedParticleNetJetTags, pfParticleNetMassRegressionJetTags,
                             pfParticleNetDiscriminatorsJetTags, pfMassDecorrelatedParticleNetDiscriminatorsJetTags)

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

_pfParticleNetMassRegressionOutputs = ['pfParticleNetMassRegressionJetTags:' + flav_name
                                       for flav_name in pfParticleNetMassRegressionJetTags.flav_names]

_pfParticleNetJetTagsAll = _pfParticleNetJetTagsProbs + _pfParticleNetJetTagsMetaDiscrs + \
    _pfMassDecorrelatedParticleNetJetTagsProbs + _pfMassDecorrelatedParticleNetJetTagsMetaDiscrs
