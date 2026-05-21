import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import pfImpactParameterTagInfos
pfImpactParameterTagInfos.jets = "akCs0PFpatJets"
pfImpactParameterTagInfos.candidates = "packedPFCandidates"
pfImpactParameterTagInfos.primaryVertex = "offlineSlimmedPrimaryVertices"
from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import pfSecondaryVertexTagInfos

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import inclusiveCandidateVertexFinder
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import candidateVertexMerger
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import candidateVertexArbitrator
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import inclusiveCandidateSecondaryVertices
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import pfInclusiveSecondaryVertexFinderTagInfos
inclusiveCandidateVertexFinder.primaryVertices  = "offlineSlimmedPrimaryVertices"
inclusiveCandidateVertexFinder.tracks= "packedPFCandidates"
inclusiveCandidateVertexFinder.minHits = 0
inclusiveCandidateVertexFinder.minPt = 0.8
candidateVertexArbitrator.tracks = "packedPFCandidates"
candidateVertexArbitrator.primaryVertices = "offlineSlimmedPrimaryVertices"

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
from RecoBTag.ImpactParameter.candidateJetProbabilityComputer_cfi import  *
from RecoBTag.ImpactParameter.pfJetProbabilityBJetTags_cfi import pfJetProbabilityBJetTags
from RecoBTag.Combined.pfDeepCSVTagInfos_cfi import pfDeepCSVTagInfos
from RecoBTag.Combined.pfDeepCSVJetTags_cfi import pfDeepCSVJetTags
pfDeepCSVTagInfos.svTagInfos = "pfSecondaryVertexTagInfos"
from RecoBTag.FeatureTools.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
pfDeepFlavourTagInfosSlimmedDeepFlavour = pfDeepFlavourTagInfos.clone(
    fallback_puppi_weight = True,
    fallback_vertex_association = True,
    jets = cms.InputTag("akCs0PFpatJets"),
    unsubjet_map = cms.InputTag("ak4PFMatchedForakCs0PFpatJets"),
    puppi_value_map = cms.InputTag(""),
    secondary_vertices = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    shallow_tag_infos = cms.InputTag("pfDeepCSVTagInfos"),
    vertex_associator = cms.InputTag(""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSCentralTagInfos,pfParticleNetFromMiniAODAK4CHSCentralJetTags,pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSForwardTagInfos,pfParticleNetFromMiniAODAK4CHSForwardJetTags,pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags

pfParticleNetFromMiniAODAK4CHSCentralTagInfosSlimmedDeepFlavour = pfParticleNetFromMiniAODAK4CHSCentralTagInfos.clone(
    jets = "akCs0PFpatJets",
    secondary_vertices = "inclusiveCandidateSecondaryVertices",
)
pfParticleNetFromMiniAODAK4CHSForwardTagInfosSlimmedDeepFlavour = pfParticleNetFromMiniAODAK4CHSCentralTagInfos.clone(
    jets = "akCs0PFpatJets",
    secondary_vertices = "inclusiveCandidateSecondaryVertices",
)

from RecoBTag.FeatureTools.pfParticleTransformerAK4TagInfos_cfi import pfParticleTransformerAK4TagInfos
pfParticleTransformerAK4TagInfosSlimmedDeepFlavour = pfParticleTransformerAK4TagInfos.clone(
    fallback_puppi_weight = True,
    fallback_vertex_association = True,
    jets = cms.InputTag("akCs0PFpatJets"),
    unsubjet_map = cms.InputTag("ak4PFMatchedForakCs0PFpatJets"),
    puppi_value_map = cms.InputTag(""),
    secondary_vertices = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    vertex_associator = cms.InputTag(""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)

from RecoBTag.ONNXRuntime.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags
pfDeepFlavourJetTagsSlimmedDeepFlavour = pfDeepFlavourJetTags.clone(src = cms.InputTag("pfDeepFlavourTagInfosSlimmedDeepFlavour"))


from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer

pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4CHSCentralTagInfosSlimmedDeepFlavour',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Central/preprocess.json',
    model_path = cms.FileInPath('RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Central/particle-net.onnx'),
    flav_names = ['probmu','probele','probtaup1h0p','probtaup1h1p','probtaup1h2p','probtaup3h0p','probtaup3h1p','probtaum1h0p','probtaum1h1p','probtaum1h2p','probtaum3h0p','probtaum3h1p','probb','probc','probuds','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

pfParticleNetFromMiniAODAK4CHSForwardJetTagsSlimmedDeepFlavour = boostedJetONNXJetTagsProducer.clone(
    src = 'pfParticleNetFromMiniAODAK4CHSForwardTagInfosSlimmedDeepFlavour',
    preprocess_json = 'RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Forward/preprocess.json',
    model_path = cms.FileInPath('RecoBTag/Combined/data/ParticleNetFromMiniAODAK4/CHS/Forward/particle-net.onnx'),
    flav_names = ['probq','probg','ptcorr','ptreshigh','ptreslow','ptnu'],
)

from RecoBTag.ONNXRuntime.pfParticleTransformerAK4JetTags_cfi import pfParticleTransformerAK4JetTags
pfParticleTransformerAK4JetTagsSlimmedDeepFlavour = pfParticleTransformerAK4JetTags.clone(src = cms.InputTag("pfParticleTransformerAK4TagInfosSlimmedDeepFlavour"))

candidateBtagging = cms.Sequence(
    pfImpactParameterTagInfos +
    pfSecondaryVertexTagInfos +
    inclusiveCandidateVertexFinder +
    candidateVertexMerger +
    candidateVertexArbitrator +
    inclusiveCandidateSecondaryVertices +
    pfInclusiveSecondaryVertexFinderTagInfos +
    pfDeepCSVTagInfos + 
    pfDeepFlavourTagInfosSlimmedDeepFlavour +
    pfParticleTransformerAK4TagInfosSlimmedDeepFlavour +
    pfJetProbabilityBJetTags +
    pfDeepCSVJetTags +
    pfDeepFlavourJetTagsSlimmedDeepFlavour +
    pfParticleTransformerAK4JetTagsSlimmedDeepFlavour
)
