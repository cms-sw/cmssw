import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.pfParticleNetAK4DiscriminatorsJetTags_cfi import pfParticleNetAK4DiscriminatorsJetTags

pfParticleNetAK4TagInfos = pfDeepBoostedJetTagInfos.clone(
    jet_radius = 0.4,
    min_jet_pt = 15,
    min_puppi_wgt = -1,
    use_puppiP4 = False,
)

pfParticleNetAK4JetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetAK4TagInfos',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/preprocess.json',
    model_path = 'RecoBTag/Combined/data/ParticleNetAK4/CHS/V00/particle-net.onnx',
    flav_names = ["probb",  "probbb",  "probc",   "probcc",  "probuds", "probg", "probundef", "probpu"],
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfParticleNetAK4Task = cms.Task(puppi, primaryVertexAssociation, pfParticleNetAK4TagInfos,
                                pfParticleNetAK4JetTags, pfParticleNetAK4DiscriminatorsJetTags)

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
