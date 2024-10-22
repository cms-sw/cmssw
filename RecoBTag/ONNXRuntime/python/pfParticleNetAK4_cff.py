import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.particleNetSonicJetTagsProducer_cfi import particleNetSonicJetTagsProducer as _particleNetSonicJetTagsProducer
from RecoBTag.ONNXRuntime.pfParticleNetAK4DiscriminatorsJetTags_cfi import pfParticleNetAK4DiscriminatorsJetTags
from Configuration.ProcessModifiers.particleNetSonicTriton_cff import particleNetSonicTriton
from Configuration.ProcessModifiers.particleNetPTSonicTriton_cff import particleNetPTSonicTriton

pfParticleNetAK4TagInfos = pfDeepBoostedJetTagInfos.clone(
    jets = "ak4PFJetsCHS",
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_puppi_wgt = -1,
    use_puppiP4 = False,
)

pfParticleNetAK4TagInfosForRECO = pfParticleNetAK4TagInfos.clone(
    min_jet_pt = 25,
    max_jet_eta = 2.5,
)

pfParticleNetAK4JetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetAK4TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/modelfile/model.onnx',
    flav_names = ["probb",  "probbb",  "probc",   "probcc",  "probuds", "probg", "probundef", "probpu"],
)

pfParticleNetAK4JetTagsForRECO = pfParticleNetAK4JetTags.clone(
    src = 'pfParticleNetAK4TagInfosForRECO',
)

pfParticleNetAK4DiscriminatorsJetTagsForRECO = pfParticleNetAK4DiscriminatorsJetTags.clone()
for discriminator in pfParticleNetAK4DiscriminatorsJetTagsForRECO.discriminators:
    for num in discriminator.numerator:
        num.setModuleLabel("pfParticleNetAK4JetTagsForRECO");
    for den in discriminator.denominator:
        den.setModuleLabel("pfParticleNetAK4JetTagsForRECO");

particleNetSonicTriton.toReplaceWith(pfParticleNetAK4JetTags, _particleNetSonicJetTagsProducer.clone(
    src = 'pfParticleNetAK4TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess.json',
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("particlenet_AK4"),
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK4/config.pbtxt"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    flav_names = pfParticleNetAK4JetTags.flav_names,
))

(particleNetSonicTriton & particleNetPTSonicTriton).toModify(pfParticleNetAK4JetTags,
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess_PT.json',
    Client = dict(
        modelName = "particlenet_AK4_PT",
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particlenet_AK4_PT/config.pbtxt"),
    )
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfParticleNetAK4Task = cms.Task(puppi, primaryVertexAssociation, pfParticleNetAK4TagInfos,
                                pfParticleNetAK4JetTags, pfParticleNetAK4DiscriminatorsJetTags)
pfParticleNetAK4TaskForRECO = cms.Task(puppi, primaryVertexAssociation, pfParticleNetAK4TagInfosForRECO,
                                pfParticleNetAK4JetTagsForRECO, pfParticleNetAK4DiscriminatorsJetTagsForRECO)

# declare all the discriminators
# probs
_pfParticleNetAK4JetTagsProbs = ['pfParticleNetAK4JetTags:' + flav_name
                                 for flav_name in pfParticleNetAK4JetTags.flav_names]
# meta-taggers
_pfParticleNetAK4JetTagsMetaDiscrs = ['pfParticleNetAK4DiscriminatorsJetTags:' + disc.name.value()
                                      for disc in pfParticleNetAK4DiscriminatorsJetTags.discriminators]
_pfParticleNetAK4JetTagsAll = _pfParticleNetAK4JetTagsProbs + _pfParticleNetAK4JetTagsMetaDiscrs


# === Negative tags ===
pfNegativeParticleNetAK4TagInfos = pfParticleNetAK4TagInfos.clone(
    flip_ip_sign = True,
    sip3dSigMax = 10,
    secondary_vertices = 'inclusiveCandidateNegativeSecondaryVertices',
)

pfNegativeParticleNetAK4JetTags = pfParticleNetAK4JetTags.clone(
    src = 'pfNegativeParticleNetAK4TagInfos',
)

# probs
_pfNegativeParticleNetAK4JetTagsProbs = ['pfNegativeParticleNetAK4JetTags:' + flav_name 
                                         for flav_name in pfNegativeParticleNetAK4JetTags.flav_names]
