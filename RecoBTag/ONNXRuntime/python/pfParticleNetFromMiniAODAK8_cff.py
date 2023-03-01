import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.FeatureTools.ParticleNetFeatureEvaluator_cfi import ParticleNetFeatureEvaluator
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8DiscriminatorsJetTags_cfi import pfParticleNetFromMiniAODAK8DiscriminatorsJetTags

pfParticleNetFromMiniAODAK8TagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJetsAK8",
    jet_radius = 0.8,
    min_jet_pt = 180,
    min_jet_eta = 0.,
    max_jet_eta = 2.5,
)


pfParticleNetFromMiniAODAK8JetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK8TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK8/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK8/particle-net.onnx',
    flav_names = ['probHtt','probHtm','probHte','probHbb', 'probHcc', 'probHqq', 'probHgg','probQCD2hf','probQCD1hf','probQCD0hf','masscorr'],
)


pfParticleNetFromMiniAODAK8Task = cms.Task( pfParticleNetFromMiniAODAK8TagInfos, pfParticleNetFromMiniAODAK8JetTags)

# declare all the discriminators
# probs
_pfParticleNetFromMiniAODAK8JetTagsProbs = ['pfParticleNetFromMiniAODAK8JetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK8JetTags.flav_names]
_pfParticleNetFromMiniAODAK8JetTagsMetaDiscr = ['pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:' + disc.name.value()
                                 for disc in pfParticleNetFromMiniAODAK8DiscriminatorsJetTags.discriminators]

_pfParticleNetFromMiniAODAK8JetTagsAll = _pfParticleNetFromMiniAODAK8JetTagsProbs + _pfParticleNetFromMiniAODAK8JetTagsMetaDiscr
