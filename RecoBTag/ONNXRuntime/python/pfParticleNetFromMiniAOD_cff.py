import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
#from RecoBTag.ONNXRuntime.pfParticleNetAK4DiscriminatorsJetTags_cfi import pfParticleNetAK4DiscriminatorsJetTags
from RecoBTag.FeatureTools.ParticleNetFeatureEvaluator_cfi import ParticleNetFeatureEvaluator

pfParticleNetFromMiniAODAK8TagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJetsAK8",
    jet_radius = 0.8,
    min_jet_pt = 200,
    min_jet_eta = 0.,
    max_jet_eta = 2.5,
    min_pt_for_track_properties = 0.95,
)


pfParticleNetFromMiniAODAK8JetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK8TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK8/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK8/modelfile/model.onnx',
    flav_names = ['probHtt','probHtm','probHte','probHbb', 'probHcc', 'probHqq', 'probHgg','probQCD2hf','probQCD1hf','probQCD0hf','masscorr'],
)


pfParticleNetFromMiniAODAK8Task = cms.Task( pfParticleNetFromMiniAODAK8TagInfos, pfParticleNetFromMiniAODAK8JetTags)

# declare all the discriminators
# probs
_pfParticleNetFromMiniAODAK8JetTagsProbs = ['pfParticleNetFromMiniAODAK8JetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK8JetTags.flav_names]
