import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfGlobalParticleTransformerAK8TagInfos_cfi import pfGlobalParticleTransformerAK8TagInfos as _pfGlobalParticleTransformerAK8TagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer

pfGlobalParticleTransformerAK8TagInfos = _pfGlobalParticleTransformerAK8TagInfos.clone(
    use_puppiP4 = False
)

pfGlobalParticleTransformerAK8JetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfGlobalParticleTransformerAK8TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/GlobalParticleTransformerAK8/PUPPI/V03/preprocess.json',
    model_path = 'RecoBTag/Combined/data/GlobalParticleTransformerAK8/PUPPI/V03/model.onnx',
    flav_names = [
        'probXbb', 'probXcc', 'probXcs', 'probXqq', 'probXtauhtaue', 'probXtauhtaum', 'probXtauhtauh', 'probXWW4q', 'probXWW3q', 'probXWWqqev', 'probXWWqqmv', 'probTopbWqq', 'probTopbWq', 'probTopbWev', 'probTopbWmv', 'probTopbWtauhv', 'probQCD', 'massCorrX2p', 'massCorrGeneric', 'probWithMassTopvsQCD', 'probWithMassWvsQCD', 'probWithMassZvsQCD'
    ] + ['hidNeuron' + str(i).zfill(3) for i in range(256)],
    debugMode = False,
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfGlobalParticleTransformerAK8Task = cms.Task(puppi, primaryVertexAssociation, pfGlobalParticleTransformerAK8TagInfos, pfGlobalParticleTransformerAK8JetTags)

# declare all the discriminators

# probs
_pfGlobalParticleTransformerAK8JetTagsProbs = ['pfGlobalParticleTransformerAK8JetTags:' + flav_name for flav_name in pfGlobalParticleTransformerAK8JetTags.flav_names]

# meta-taggers
_pfGlobalParticleTransformerAK8JetTagsMetaDiscrs = []

_pfGlobalParticleTransformerAK8JetTagsAll = _pfGlobalParticleTransformerAK8JetTagsProbs + _pfGlobalParticleTransformerAK8JetTagsMetaDiscrs
