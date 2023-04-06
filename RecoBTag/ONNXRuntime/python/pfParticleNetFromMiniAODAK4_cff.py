import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.FeatureTools.ParticleNetFeatureEvaluator_cfi import ParticleNetFeatureEvaluator
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4DiscriminatorsJetTags_cfi import *

pfParticleNetFromMiniAODAK4CHSCentralTagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJets",
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_jet_eta = 0.,
    max_jet_eta = 2.5,
)

pfParticleNetFromMiniAODAK4CHSForwardTagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJets",
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_jet_eta = 2.5,
    max_jet_eta = 4.7,
)

pfParticleNetFromMiniAODAK4PuppiCentralTagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJetsPuppi",
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_jet_eta = 0.,
    max_jet_eta = 2.5,
)

pfParticleNetFromMiniAODAK4PuppiForwardTagInfos = ParticleNetFeatureEvaluator.clone(
    jets = "slimmedJetsPuppi",
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_jet_eta = 2.5,
    max_jet_eta = 4.7,
)


pfParticleNetFromMiniAODAK4CHSCentralJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4CHSCentralTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Central/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Central/particle-net.onnx',
    flav_names = ['probmu','probele','probtaup1h0p','probtaup1h1p','probtaup1h2p','probtaup3h0p','probtaup3h1p','probtaum1h0p','probtaum1h1p','probtaum1h2p','probtaum3h0p','probtaum3h1p','probb','probc','probuds','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

pfParticleNetFromMiniAODAK4CHSForwardJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4CHSForwardTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Forward/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Forward/particle-net.onnx',
    flav_names = ['probq','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

pfParticleNetFromMiniAODAK4PuppiCentralJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4PuppiCentralTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/PUPPI/Central/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/PUPPI/Central/particle-net.onnx',
    flav_names = ['probmu','probele','probtaup1h0p','probtaup1h1p','probtaup1h2p','probtaup3h0p','probtaup3h1p','probtaum1h0p','probtaum1h1p','probtaum1h2p','probtaum3h0p','probtaum3h1p','probb','probc','probuds','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

pfParticleNetFromMiniAODAK4PuppiForwardJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4PuppiForwardTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/PUPPI/Forward/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/PUPPI/Forward/particle-net.onnx',
    flav_names = ['probq','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

pfParticleNetFromMiniAODAK4CHSTask = cms.Task( pfParticleNetFromMiniAODAK4CHSCentralTagInfos, pfParticleNetFromMiniAODAK4CHSForwardTagInfos, pfParticleNetFromMiniAODAK4CHSCentralJetTags, pfParticleNetFromMiniAODAK4CHSForwardJetTags)
pfParticleNetFromMiniAODAK4PuppiTask = cms.Task( pfParticleNetFromMiniAODAK4PuppiCentralTagInfos, pfParticleNetFromMiniAODAK4PuppiForwardTagInfos, pfParticleNetFromMiniAODAK4PuppiCentralJetTags, pfParticleNetFromMiniAODAK4PuppiForwardJetTags)

# declare all the discriminators
# probs
_pfParticleNetFromMiniAODAK4CHSCentralJetTagsProbs = ['pfParticleNetFromMiniAODAK4CHSCentralJetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK4CHSCentralJetTags.flav_names]
_pfParticleNetFromMiniAODAK4CHSForwardJetTagsProbs = ['pfParticleNetFromMiniAODAK4CHSForwardJetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK4CHSForwardJetTags.flav_names]
_pfParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs = ['pfParticleNetFromMiniAODAK4PuppiCentralJetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK4PuppiCentralJetTags.flav_names]
_pfParticleNetFromMiniAODAK4PuppiForwardJetTagsProbs = ['pfParticleNetFromMiniAODAK4PuppiForwardJetTags:' + flav_name
                                 for flav_name in pfParticleNetFromMiniAODAK4PuppiForwardJetTags.flav_names]

_pfParticleNetFromMiniAODAK4CHSCentralJetTagsMetaDiscr = ['pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:' + disc.name.value()
                                 for disc in pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags.discriminators]
_pfParticleNetFromMiniAODAK4CHSForwardJetTagsMetaDiscr = ['pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags:' + disc.name.value()
                                 for disc in pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags.discriminators]
_pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr = ['pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:' + disc.name.value()
                                 for disc in pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags.discriminators]
_pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr = ['pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags:' + disc.name.value()
                                 for disc in pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags.discriminators]

_pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll = _pfParticleNetFromMiniAODAK4CHSCentralJetTagsProbs + _pfParticleNetFromMiniAODAK4CHSCentralJetTagsMetaDiscr
_pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll = _pfParticleNetFromMiniAODAK4CHSForwardJetTagsProbs + _pfParticleNetFromMiniAODAK4CHSForwardJetTagsMetaDiscr
_pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll = _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsProbs + _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr
_pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll = _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsProbs + _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr
