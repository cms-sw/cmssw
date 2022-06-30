import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.particleNetSonicJetTagsProducer_cfi import particleNetSonicJetTagsProducer as _particleNetSonicJetTagsProducer
from RecoBTag.ONNXRuntime.pfParticleNetDiscriminatorsJetTags_cfi import pfParticleNetDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfMassDecorrelatedParticleNetDiscriminatorsJetTags_cfi import pfMassDecorrelatedParticleNetDiscriminatorsJetTags
from Configuration.ProcessModifiers.particleNetSonicTriton_cff import particleNetSonicTriton
from Configuration.ProcessModifiers.particleNetPTSonicTriton_cff import particleNetPTSonicTriton

pfParticleNetTagInfos = pfDeepBoostedJetTagInfos.clone(
    use_puppiP4 = False
)

pfParticleNetJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/modelfile/model.onnx',
    flav_names = ["probTbcq",  "probTbqq",  "probTbc",   "probTbq",  "probTbel", "probTbmu", "probTbta",
                  "probWcq",   "probWqq",   "probZbb",   "probZcc",  "probZqq",  "probHbb", "probHcc",
                  "probHqqqq", "probQCDbb", "probQCDcc", "probQCDb", "probQCDc", "probQCDothers"],
)

particleNetSonicTriton.toReplaceWith(pfParticleNetJetTags, _particleNetSonicJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/preprocess.json',
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("particlenet"),
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet/config.pbtxt"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    flav_names = pfParticleNetJetTags.flav_names,
))

(particleNetSonicTriton & particleNetPTSonicTriton).toModify(pfParticleNetJetTags,
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/General/V01/preprocess_PT.json',
    Client = dict(
        modelName = "particlenet_PT",
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_PT/config.pbtxt"),
    )
)

pfMassDecorrelatedParticleNetJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/modelfile/model.onnx',
    flav_names = ["probXbb", "probXcc", "probXqq", "probQCDbb", "probQCDcc",
                  "probQCDb", "probQCDc", "probQCDothers"],
)

particleNetSonicTriton.toReplaceWith(pfMassDecorrelatedParticleNetJetTags, _particleNetSonicJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/preprocess.json',
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        modelName = cms.string("particlenet_AK8_MD-2prong"),
        mode = cms.string("Async"),
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK8_MD-2prong/config.pbtxt"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
    ),
    flav_names = pfMassDecorrelatedParticleNetJetTags.flav_names,
))

(particleNetSonicTriton & particleNetPTSonicTriton).toModify(pfMassDecorrelatedParticleNetJetTags,
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MD-2prong/V01/preprocess_PT.json',
    Client = dict(
        modelName = "particlenet_AK8_MD-2prong_PT",
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK8_MD-2prong_PT/config.pbtxt"),
    )
)

pfParticleNetMassRegressionJetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/modelfile/model.onnx',
    flav_names = ["mass"],
)

particleNetSonicTriton.toReplaceWith(pfParticleNetMassRegressionJetTags, _particleNetSonicJetTagsProducer.clone(
    src = 'pfParticleNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/preprocess.json',
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        modelName = cms.string("particlenet_AK8_MassRegression"),
        mode = cms.string("Async"),
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK8_MassRegression/config.pbtxt"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
    ),
    flav_names = pfParticleNetMassRegressionJetTags.flav_names,
))

(particleNetSonicTriton & particleNetPTSonicTriton).toModify(pfParticleNetMassRegressionJetTags,
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK8/MassRegression/V01/preprocess_PT.json',
    Client = dict(
        modelName = "particlenet_AK8_MassRegression_PT",
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK8_MassRegression_PT/config.pbtxt"),
    )
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

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
