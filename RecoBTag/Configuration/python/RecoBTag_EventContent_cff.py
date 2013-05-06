import FWCore.ParameterSet.Config as cms

#Full Event content
RecoBTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_impactParameterTagInfos_*_*',
        'keep *_trackCountingHighEffBJetTags_*_*',
        'keep *_trackCountingHighPurBJetTags_*_*',
        'keep *_jetProbabilityBJetTags_*_*',
        'keep *_jetBProbabilityBJetTags_*_*',
        'keep *_secondaryVertexTagInfos_*_*',
        'keep *_ghostTrackVertexTagInfos_*_*',
        'keep *_simpleSecondaryVertexBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighPurBJetTags_*_*',
        'keep *_combinedSecondaryVertexBJetTags_*_*',
        'keep *_combinedSecondaryVertexMVABJetTags_*_*',
        'keep *_ghostTrackBJetTags_*_*',
        'keep *_softPFMuonsTagInfos_*_*',
        'keep *_softPFElectronsTagInfos_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_softMuonTagInfos_*_*',
        'keep *_softMuonBJetTags_*_*',
        'keep *_softMuonByIP3dBJetTags_*_*',
        'keep *_softMuonByPtBJetTags_*_*',
    )
)
#RECO content
RecoBTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_impactParameterTagInfos_*_*',
        'keep *_trackCountingHighEffBJetTags_*_*',
        'keep *_trackCountingHighPurBJetTags_*_*',
        'keep *_jetProbabilityBJetTags_*_*',
        'keep *_jetBProbabilityBJetTags_*_*',
        'keep *_secondaryVertexTagInfos_*_*',
        'keep *_ghostTrackVertexTagInfos_*_*',
        'keep *_simpleSecondaryVertexBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighPurBJetTags_*_*',
        'keep *_combinedSecondaryVertexBJetTags_*_*',
        'keep *_combinedSecondaryVertexMVABJetTags_*_*',
        'keep *_ghostTrackBJetTags_*_*',
        'keep *_softPFMuonsTagInfos_*_*',
        'keep *_softPFElectronsTagInfos_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_softMuonTagInfos_*_*',
        'keep *_softMuonBJetTags_*_*',
        'keep *_softMuonByIP3dBJetTags_*_*',
        'keep *_softMuonByPtBJetTags_*_*',
    )
)
#AOD content
RecoBTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
#        'keep *_impactParameterTagInfos_*_*',
        'keep *_trackCountingHighEffBJetTags_*_*',
        'keep *_trackCountingHighPurBJetTags_*_*',
        'keep *_jetProbabilityBJetTags_*_*',
        'keep *_jetBProbabilityBJetTags_*_*',
#        'keep *_secondaryVertexTagInfos_*_*',
#        'keep *_ghostTrackVertexTagInfos_*_*',
        'keep *_simpleSecondaryVertexBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_simpleSecondaryVertexHighPurBJetTags_*_*',
        'keep *_combinedSecondaryVertexBJetTags_*_*',
        'keep *_combinedSecondaryVertexMVABJetTags_*_*',
        'keep *_ghostTrackBJetTags_*_*',
#        'keep *_btagSoftElectrons_*_*',
##        'keep *_softElectronCands_*_*',
##        'keep *_softPFElectrons_*_*',
#       'keep *_softElectronTagInfos_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
#        'keep *_softMuonTagInfos_*_*',
        'keep *_softMuonBJetTags_*_*',
        'keep *_softMuonByIP3dBJetTags_*_*',
        'keep *_softMuonByPtBJetTags_*_*',
    )
)
