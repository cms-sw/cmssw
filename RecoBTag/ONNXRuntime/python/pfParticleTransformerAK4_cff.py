import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfParticleTransformerAK4TagInfos_cfi import pfParticleTransformerAK4TagInfos

from RecoBTag.ONNXRuntime.pfParticleTransformerAK4JetTags_cfi import pfParticleTransformerAK4JetTags
from RecoBTag.ONNXRuntime.pfParticleTransformerAK4DiscriminatorsJetTags_cfi import pfParticleTransformerAK4DiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleTransformerAK4SonicJetTags_cfi import pfParticleTransformerAK4SonicJetTags as _pfParticleTransformerAK4SonicJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation
from Configuration.ProcessModifiers.particleTransformerAK4SonicTriton_cff import particleTransformerAK4SonicTriton


particleTransformerAK4SonicTriton.toReplaceWith(pfParticleTransformerAK4JetTags, _pfParticleTransformerAK4SonicJetTags.clone(
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("particletransformer_AK4"), # double check
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/particletransformer_AK4/config.pbtxt"), # this is SONIC, not currently in the CMSSW, so the models/ will be copied to this location privately
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    flav_names = pfParticleTransformerAK4JetTags.flav_names,
))

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
