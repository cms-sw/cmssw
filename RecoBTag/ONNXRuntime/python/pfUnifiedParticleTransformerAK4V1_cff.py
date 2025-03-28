import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfUnifiedParticleTransformerAK4TagInfos_cfi import pfUnifiedParticleTransformerAK4TagInfos
from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4V1DiscriminatorsJetTags_cfi import pfUnifiedParticleTransformerAK4V1DiscriminatorsJetTags

from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4JetTags_cfi import pfUnifiedParticleTransformerAK4JetTags as _pfUnifiedParticleTransformerAK4JetTags

#
pfUnifiedParticleTransformerAK4V1TagInfos = pfUnifiedParticleTransformerAK4TagInfos.clone()


pfUnifiedParticleTransformerAK4V1JetTags = _pfUnifiedParticleTransformerAK4JetTags.clone(
   src = "pfUnifiedParticleTransformerAK4V1TagInfos",
   model_path = cms.FileInPath('RecoBTag/Combined/data/UParTAK4/PUPPI/V00/UParTAK4.onnx'),
 )

# declare all the discriminators
# probs
_pfUnifiedParticleTransformerAK4V1JetTagsProbs = ['pfUnifiedParticleTransformerAK4V1JetTags:' + flav_name
                                 for flav_name in pfUnifiedParticleTransformerAK4V1JetTags.flav_names]
# meta-taggers
_pfUnifiedParticleTransformerAK4V1JetTagsMetaDiscrs = ['pfUnifiedParticleTransformerAK4V1DiscriminatorsJetTags:' + disc.name.value()
                                      for disc in pfUnifiedParticleTransformerAK4V1DiscriminatorsJetTags.discriminators]

_pfUnifiedParticleTransformerAK4V1JetTagsAll = _pfUnifiedParticleTransformerAK4V1JetTagsProbs + _pfUnifiedParticleTransformerAK4V1JetTagsMetaDiscrs

# run from MiniAOD instead
pfUnifiedParticleTransformerAK4V1FromMiniAODTask = cms.Task(pfUnifiedParticleTransformerAK4V1TagInfos,
  pfUnifiedParticleTransformerAK4V1JetTags,
  pfUnifiedParticleTransformerAK4V1DiscriminatorsJetTags
)
