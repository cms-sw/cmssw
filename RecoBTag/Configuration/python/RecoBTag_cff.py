import FWCore.ParameterSet.Config as cms

# define the b-tag squences for offline reconstruction
from RecoBTag.SoftLepton.softLepton_cff import *
from RecoBTag.ImpactParameter.impactParameter_cff import *
from RecoBTag.SecondaryVertex.secondaryVertex_cff import *
from RecoBTag.Combined.combinedMVA_cff import *
from RecoBTag.CTagging.RecoCTagging_cff import *
from RecoBTag.Combined.deepFlavour_cff import *
from RecoBTag.ONNXRuntime.pfDeepFlavour_cff import *
from RecoBTag.ONNXRuntime.pfDeepDoubleX_cff import *
from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import *
from RecoBTag.ONNXRuntime.pfHiggsInteractionNet_cff import *
from RecoBTag.ONNXRuntime.pfParticleNet_cff import *
from RecoBTag.ONNXRuntime.pfParticleNetAK4_cff import *
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import *
from RecoBTag.PixelCluster.pixelClusterTagInfos_cfi import *

legacyBTaggingTask = cms.Task(
    # impact parameters and IP-only algorithms
    impactParameterTagInfos,
    trackCountingHighEffBJetTags,
    jetProbabilityBJetTags,
    jetBProbabilityBJetTags,

    # SV tag infos depending on IP tag infos, and SV (+IP) based algos
    secondaryVertexTagInfos,
    simpleSecondaryVertexHighEffBJetTags,
    combinedSecondaryVertexV2BJetTags,
    inclusiveSecondaryVertexFinderTagInfos,
    combinedInclusiveSecondaryVertexV2BJetTags,
    ghostTrackVertexTagInfos,
    ghostTrackBJetTags,

    # soft lepton tag infos and algos
    softPFMuonsTagInfos,
    softPFMuonBJetTags,
    softPFElectronsTagInfos,
    softPFElectronBJetTags,

    # overall combined taggers
    combinedMVAV2BJetTags,
    
    # pixel cluster
    pixelClusterTagInfos,
)
legacyBTagging = cms.Sequence(legacyBTaggingTask)

# new candidate-based fwk, with PF inputs
pfBTaggingTask = cms.Task(
    # impact parameters and IP-only algorithms
    pfImpactParameterTagInfos,
    pfTrackCountingHighEffBJetTags,
    pfJetProbabilityBJetTags,
    pfJetBProbabilityBJetTags,

    # SV tag infos depending on IP tag infos, and SV (+IP) based algos
    pfSecondaryVertexTagInfos,
    pfSimpleSecondaryVertexHighEffBJetTags,
    pfCombinedSecondaryVertexV2BJetTags,
    inclusiveCandidateVertexingTask,
    pfInclusiveSecondaryVertexFinderTagInfos,
    pfSimpleInclusiveSecondaryVertexHighEffBJetTags,
    pfCombinedInclusiveSecondaryVertexV2BJetTags,
    pfGhostTrackVertexTagInfos,
    pfGhostTrackBJetTags,
    pfDeepCSVTask,

    # soft lepton tag infos and algos
    softPFMuonsTagInfos,
    softPFMuonBJetTags,
    softPFElectronsTagInfos,
    softPFElectronBJetTags,

    # overall combined taggers
    #CSV + soft-lepton + jet probability discriminators combined
    pfCombinedMVAV2BJetTags,
    pfChargeBJetTags,
    
    # pixel cluster
    pixelClusterTagInfos,

)

pfBTagging = cms.Sequence(pfBTaggingTask)

btaggingTask = cms.Task(
    pfBTaggingTask,
    pfCTaggingTask
)
btagging = cms.Sequence(btaggingTask)

## modifying b-tagging task in Run3 adding ParticleNet inferece
from Configuration.Eras.Modifier_run3_common_cff import run3_common
_pfBTaggingTask_run3 = cms.Task(
    # Keep all the infos and DeepCSV
    pfImpactParameterTagInfos,
    pfTrackCountingHighEffBJetTags,
    pfJetProbabilityBJetTags,
    pfJetBProbabilityBJetTags,

    pfSecondaryVertexTagInfos,
    inclusiveCandidateVertexingTask,
    pfInclusiveSecondaryVertexFinderTagInfos,
    pfGhostTrackVertexTagInfos,
    pfDeepCSVTask,

    softPFMuonsTagInfos,
    softPFElectronsTagInfos,
    pixelClusterTagInfos,

    pfParticleNetAK4TaskForRECO,
    pfParticleNetTask
)
_pfCTaggingTask_run3 = cms.Task(
    inclusiveCandidateVertexingCvsLTask,
    pfInclusiveSecondaryVertexFinderCvsLTagInfos,
)
run3_common.toReplaceWith( pfBTaggingTask, _pfBTaggingTask_run3 )
run3_common.toReplaceWith( pfCTaggingTask, _pfCTaggingTask_run3 )

