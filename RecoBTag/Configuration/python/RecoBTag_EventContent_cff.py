import FWCore.ParameterSet.Config as cms

#Full Event content
RecoBTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_softPFMuonsTagInfos_*_*',
        'keep *_softPFElectronsTagInfos_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_pfImpactParameterTagInfos_*_*',
        'keep *_pfTrackCountingHighEffBJetTags_*_*',
        'keep *_pfJetProbabilityBJetTags_*_*',
        'keep *_pfJetBProbabilityBJetTags_*_*',
        'keep *_pfSecondaryVertexTagInfos_*_*',
        'keep *_pfInclusiveSecondaryVertexFinderTagInfos_*_*',
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*',
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*',
        'keep *_pfGhostTrackVertexTagInfos_*_*',
        'keep *_pfGhostTrackBJetTags_*_*',
        'keep *_pfCombinedMVAV2BJetTags_*_*',
        'keep *_inclusiveCandidateSecondaryVertices_*_*',
        # CTagging
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*',
        'keep *_pfInclusiveSecondaryVertexFinderCvsLTagInfos_*_*',
        'keep *_pfCombinedCvsLJetTags_*_*',
        'keep *_pfCombinedCvsBJetTags_*_*',
        # ChargeTag
        'keep *_pfChargeBJetTags_*_*',
        # DeepFlavour
        'keep *_pfDeepCSVJetTags_*_*',
        # DeepCMVA
        'keep *_pfDeepCMVAJetTags_*_*',
    )
)
#RECO content
RecoBTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_softPFMuonsTagInfos_*_*',
        'keep *_softPFElectronsTagInfos_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_pfImpactParameterTagInfos_*_*',
        'keep *_pfTrackCountingHighEffBJetTags_*_*',
        'keep *_pfJetProbabilityBJetTags_*_*',
        'keep *_pfJetBProbabilityBJetTags_*_*',
        'keep *_pfSecondaryVertexTagInfos_*_*',
        'keep *_pfInclusiveSecondaryVertexFinderTagInfos_*_*',
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*',
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*',
        'keep *_pfGhostTrackVertexTagInfos_*_*',
        'keep *_pfGhostTrackBJetTags_*_*',
        'keep *_pfCombinedMVAV2BJetTags_*_*',
        'keep *_inclusiveCandidateSecondaryVertices_*_*',
        # CTagging
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*',
        'keep *_pfInclusiveSecondaryVertexFinderCvsLTagInfos_*_*',
        'keep *_pfCombinedCvsLJetTags_*_*',
        'keep *_pfCombinedCvsBJetTags_*_*',
        # ChargeTag
        'keep *_pfChargeBJetTags_*_*',
        # DeepFlavour
        'keep *_pfDeepCSVJetTags_*_*',
        # DeepCMVA
        'keep *_pfDeepCMVAJetTags_*_*',
    )
)
#AOD content
RecoBTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_pfTrackCountingHighEffBJetTags_*_*',
        'keep *_pfJetProbabilityBJetTags_*_*',
        'keep *_pfJetBProbabilityBJetTags_*_*',
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*',
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*',
        'keep *_pfGhostTrackBJetTags_*_*',
        'keep *_pfCombinedMVAV2BJetTags_*_*',
        'keep *_inclusiveCandidateSecondaryVertices_*_*',
        # CTagging
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*',
        'keep *_pfCombinedCvsLJetTags_*_*',
        'keep *_pfCombinedCvsBJetTags_*_*',
        # ChargeTag
        'keep *_pfChargeBJetTags_*_*',
        # DeepFlavour
        'keep *_pfDeepCSVJetTags_*_*',
        # DeepCMVA
        'keep *_pfDeepCMVAJetTags_*_*',
    )
)
