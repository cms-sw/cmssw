# Unit test configuration file for RecoBTagInfo producers:
# Verify the use of unsubtracted jet map
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring("root://xrootd-cms.infn.it//store/hidata/HIRun2023A/HIPhysicsRawPrime0/MINIAOD/PromptReco-v2/000/375/823/00000/8158260e-df3c-45a5-a121-55345a682a23.root")
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '132X_dataRun3_Prompt_v4', '')

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
process.ak4PFJets = ak4PFJets.clone(rParam = 0.4, src = 'packedPFCandidates')

process.unsubJets = cms.EDProducer("JetMatcherDR",
    source = cms.InputTag("ak4PFJets"),
    matched = cms.InputTag("ak4PFJets")
)

from RecoBTag.FeatureTools.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
process.pfDeepFlavourTagInfos = pfDeepFlavourTagInfos.clone(
    jets = "ak4PFJets",
    unsubjet_map = "unsubJets",
    fallback_puppi_weight = True,
    fallback_vertex_association = True,
    puppi_value_map = "",
    secondary_vertices = "inclusiveCandidateSecondaryVertices",
    shallow_tag_infos = "pfDeepCSVTagInfos",
    vertex_associator = "",
    vertices = "offlineSlimmedPrimaryVertices"
)

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
process.pfDeepBoostedJetTagInfos = pfDeepBoostedJetTagInfos.clone(
    jets = "ak4PFJets",
    unsubjet_map = "unsubJets",
    use_puppiP4 = False,
    puppi_value_map = "",
    secondary_vertices = "inclusiveCandidateSecondaryVertices",
    vertex_associator = "",
    vertices = "offlineSlimmedPrimaryVertices",
    pf_candidates = "packedPFCandidates"
)

from RecoBTag.FeatureTools.pfParticleTransformerAK4TagInfos_cfi import pfParticleTransformerAK4TagInfos
process.pfParticleTransformerAK4TagInfos = pfParticleTransformerAK4TagInfos.clone(
    jets = "ak4PFJets",
    unsubjet_map = "unsubJets",
    fallback_puppi_weight = True,
    fallback_vertex_association = True,
    puppi_value_map = "",
    secondary_vertices = "inclusiveCandidateSecondaryVertices",
    vertex_associator = "",
    vertices = "offlineSlimmedPrimaryVertices"
)

process.p = cms.Path(process.ak4PFJets *
                     process.unsubJets *
                     process.pfDeepFlavourTagInfos *
                     process.pfDeepBoostedJetTagInfos *
                     process.pfParticleTransformerAK4TagInfos)
