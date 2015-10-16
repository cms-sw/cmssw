import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*_*_*', 
                                           'keep recoPFJets_*_*_*',
                                           'keep recoTrackJets_*_*_*',
                                           'keep recoJPTJets_*_*_*',
                                           'keep recoBasicJets_*_*_*',
                                           'keep *_caloTowers_*_*', 
                                           'keep *_towerMaker_*_*',
                                           'keep *_CastorTowerReco_*_*',                                           
                                           'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',
                                           'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
                                           'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
                                           'keep *_kt4JetExtender_*_*',
                                           'keep *_ak4JetTracksAssociatorAtVertex*_*_*', 
                                           'keep *_ak4JetTracksAssociatorAtCaloFace*_*_*', 
                                           'keep *_ak4JetExtender_*_*',
                                           'keep *_ak4JetTracksAssociatorExplicit_*_*',
                                           'keep *_ak7JetTracksAssociatorAtVertex*_*_*', 
                                           'keep *_ak7JetTracksAssociatorAtCaloFace*_*_*', 
                                           'keep *_ak7JetExtender_*_*',
                                           'keep *_*JetID_*_*',
                                           #keep jet area variables for jet colls in RECO 
                                           'keep *_kt4CaloJets_*_*', 
                                           'keep *_kt6CaloJets_*_*',
                                           'keep *_ak4CaloJets_*_*',
                                           'keep *_ak5CaloJets_*_*',
                                           'keep *_ak7CaloJets_*_*',
                                           'keep *_kt4PFJets_*_*', 
                                           'keep *_kt6PFJets_*_*',
                                           'keep *_ak4PFJets_*_*',
                                           'keep *_ak5PFJets_*_*',
                                           'keep *_ak7PFJets_*_*',
                                           'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*',
                                           'keep *_ak4TrackJets_*_*',
                                           'keep *_kt4TrackJets_*_*',
                                           'keep *_ak5CastorJets_*_*',
                                           'keep *_ak5CastorJetID_*_*',
                                           'keep *_ak7CastorJets_*_*',
                                           'keep *_ak7CastorJetID_*_*',
                                           'keep *_fixedGridRho*_*_*',
                                           'keep *_ca*Mass_*_*',
                                           'keep *_ak*Mass_*_*'
        )
)
RecoGenJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenJets_*_*_*', 
                                           'keep *_genParticle_*_*')
)
#RECO content
RecoJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ak4CaloJets_*_*',
                                           'keep *_ak4PFJets_*_*',
                                           'keep *_ak4PFJetsCHS_*_*',
                                           'keep *_ak8PFJetsCHS_*_*',
                                           'keep *_ak8PFJetsCHSSoftDrop_*_*',                                           
                                           'keep *_cmsTopTagPFJetsCHS_*_*',
                                           'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*',                                  
                                           'keep *_ak4TrackJets_*_*',
                                           'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',         
                                           'keep *_caloTowers_*_*', 
                                           'keep *_towerMaker_*_*',
                                           'keep *_CastorTowerReco_*_*',                                           
                                           'keep *_ak4JetTracksAssociatorAtVertex_*_*',
                                           'keep *_ak4JetTracksAssociatorAtVertexPF_*_*',
                                           'keep *_ak4JetTracksAssociatorAtCaloFace_*_*',
                                           'keep *_ak4JetTracksAssociatorExplicit_*_*',
                                           'keep *_ak4JetExtender_*_*',
                                           'keep *_ak4JetID_*_*',
					   'keep *_ak5CastorJets_*_*',
                                           'keep *_ak5CastorJetID_*_*',
                                           'keep *_ak7CastorJets_*_*',
                                           'keep *_ak7CastorJetID_*_*',
                                           #'keep *_fixedGridRho*_*_*',
                                           'keep *_fixedGridRhoAll_*_*',
                                           'keep *_fixedGridRhoFastjetAll_*_*',
                                           'keep *_fixedGridRhoFastjetAllTmp_*_*',
                                           'keep *_fixedGridRhoFastjetAllCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralChargedPileUp_*_*',
                                           'keep *_fixedGridRhoFastjetCentralNeutral_*_*',
                                           'keep *_ak8PFJetsCHSSoftDropMass_*_*'                                 
                                           )
)
RecoGenJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ak4GenJets_*_*',
                                           'keep *_ak8GenJets_*_*',
                                           'keep *_ak4GenJetsNoNu_*_*',
                                           'keep *_ak8GenJetsNoNu_*_*',
                                           'keep *_genParticle_*_*')
    )
#AOD content
RecoJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ak4CaloJets_*_*',
                                           'keep *_ak4PFJetsCHS_*_*',
                                           'keep *_ak8PFJetsCHS_*_*',
                                           'keep *_ak8PFJetsCHSSoftDrop_*_*',
                                           'keep *_cmsTopTagPFJetsCHS_*_*',
                                           'keep *_ak4PFJets_*_*',
                                           'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*',    
                                           'keep *_ak4TrackJets_*_*',
                                           'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',                                             
                                           'keep *_caloTowers_*_*', 
                                           'keep *_CastorTowerReco_*_*',                                           
                                           'keep *_ak4JetTracksAssociatorAtVertex_*_*',
                                           'keep *_ak4JetTracksAssociatorAtVertexPF_*_*', 
                                           'keep *_ak4JetTracksAssociatorExplicit_*_*',
                                           'keep *_ak4JetExtender_*_*', 
                                           'keep *_ak4JetID_*_*',
                                           'keep *_ak5CastorJets_*_*',
                                           'keep *_ak5CastorJetID_*_*',
                                           'keep *_ak7CastorJets_*_*',
                                           'keep *_ak7CastorJetID_*_*',
                                           #'keep *_fixedGridRho*_*_*',
                                           'keep *_fixedGridRhoAll_*_*',
                                           'keep *_fixedGridRhoFastjetAll_*_*',
                                           'keep *_fixedGridRhoFastjetAllTmp_*_*',
                                           'keep *_fixedGridRhoFastjetAllCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralChargedPileUp_*_*',
                                           'keep *_fixedGridRhoFastjetCentralNeutral_*_*',
                                           'drop doubles_*Jets_rhos_*',
                                           'drop doubles_*Jets_sigmas_*',
                                           'keep *_ak8PFJetsCHSSoftDropMass_*_*'                                 
                                           )
    )
RecoGenJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ak4GenJets_*_*',
                                           'keep *_ak8GenJets_*_*',
                                           'keep *_ak4GenJetsNoNu_*_*',
                                           'keep *_ak8GenJetsNoNu_*_*',
                                           'keep *_genParticle_*_*'
                                           )
    )
