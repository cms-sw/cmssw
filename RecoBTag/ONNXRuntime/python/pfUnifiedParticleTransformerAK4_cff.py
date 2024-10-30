import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfUnifiedParticleTransformerAK4TagInfos_cfi import pfUnifiedParticleTransformerAK4TagInfos

from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4JetTags_cfi import pfUnifiedParticleTransformerAK4JetTags
from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4DiscriminatorsJetTags_cfi import pfUnifiedParticleTransformerAK4DiscriminatorsJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

# declare all the discriminators
# probs
_pfUnifiedParticleTransformerAK4JetTagsProbs = ['pfUnifiedParticleTransformerAK4JetTags:' + flav_name
                                 for flav_name in pfUnifiedParticleTransformerAK4JetTags.flav_names]
# meta-taggers
_pfUnifiedParticleTransformerAK4JetTagsMetaDiscrs = ['pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:' + disc.name.value()
                                      for disc in pfUnifiedParticleTransformerAK4DiscriminatorsJetTags.discriminators]
_pfUnifiedParticleTransformerAK4JetTagsAll = _pfUnifiedParticleTransformerAK4JetTagsProbs + _pfUnifiedParticleTransformerAK4JetTagsMetaDiscrs



# ==
# This task is not used, useful only if we run it from RECO jets (RECO/AOD)
pfUnifiedParticleTransformerAK4Task = cms.Task(puppi, primaryVertexAssociation,
                             pfUnifiedParticleTransformerAK4TagInfos, pfUnifiedParticleTransformerAK4JetTags,
                             pfUnifiedParticleTransformerAK4DiscriminatorsJetTags)
# run from MiniAOD instead
pfUnifiedParticleTransformerAK4FromMiniAODTask = cms.Task(pfUnifiedParticleTransformerAK4TagInfos,
                             pfUnifiedParticleTransformerAK4JetTags,
                             pfUnifiedParticleTransformerAK4DiscriminatorsJetTags)

# === Negative tags ===                                                                                                                                                      
pfNegativeUnifiedParticleTransformerAK4TagInfos = pfUnifiedParticleTransformerAK4TagInfos.clone(
    flip = True,
    secondary_vertices = 'inclusiveCandidateNegativeSecondaryVertices',
)
pfNegativeUnifiedParticleTransformerAK4JetTags = pfUnifiedParticleTransformerAK4JetTags.clone(
    src = 'pfNegativeParticleTransformerAK4TagInfos',
)

# probs                                                                                                                                                                      
_pfNegativeUnifiedParticleTransformerAK4JetTagsProbs = ['pfNegativeUnifiedParticleTransformerAK4JetTags:' + flav_name
                                 for flav_name in pfUnifiedParticleTransformerAK4JetTags.flav_names]
