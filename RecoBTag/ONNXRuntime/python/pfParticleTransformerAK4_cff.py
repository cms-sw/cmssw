import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfParticleTransformerAK4TagInfos_cfi import pfParticleTransformerAK4TagInfos

from RecoBTag.ONNXRuntime.pfParticleTransformerAK4JetTags_cfi import pfParticleTransformerAK4JetTags
from RecoBTag.ONNXRuntime.pfParticleTransformerAK4DiscriminatorsJetTags_cfi import pfParticleTransformerAK4DiscriminatorsJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

# declare all the discriminators
# probs
_pfParticleTransformerAK4JetTagsProbs = ['pfParticleTransformerAK4JetTags:' + flav_name
                                 for flav_name in pfParticleTransformerAK4JetTags.flav_names]
# meta-taggers
_pfParticleTransformerAK4JetTagsMetaDiscrs = ['pfParticleTransformerAK4DiscriminatorsJetTags:' + disc.name.value()
                                      for disc in pfParticleTransformerAK4DiscriminatorsJetTags.discriminators]
_pfParticleTransformerAK4JetTagsAll = _pfParticleTransformerAK4JetTagsProbs + _pfParticleTransformerAK4JetTagsMetaDiscrs



# ==
# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfParticleTransformerAK4Task = cms.Task(puppi, primaryVertexAssociation,
                             pfParticleTransformerAK4TagInfos, pfParticleTransformerAK4JetTags,
                             pfParticleTransformerAK4DiscriminatorsJetTags)
# run from MiniAOD instead
pfParticleTransformerAK4FromMiniAODTask = cms.Task(pfParticleTransformerAK4TagInfos,
                             pfParticleTransformerAK4JetTags,
                             pfParticleTransformerAK4DiscriminatorsJetTags)

# === Negative tags ===
pfNegativeParticleTransformerAK4TagInfos = pfParticleTransformerAK4TagInfos.clone(
    flip = True,
    secondary_vertices = 'inclusiveCandidateNegativeSecondaryVertices',
)
pfNegativeParticleTransformerAK4JetTags = pfParticleTransformerAK4JetTags.clone(
    src = 'pfNegativeParticleTransformerAK4TagInfos',
)

# probs
_pfNegativeParticleTransformerAK4JetTagsProbs = ['pfNegativeParticleTransformerAK4JetTags:' + flav_name
                                 for flav_name in pfParticleTransformerAK4JetTags.flav_names]
