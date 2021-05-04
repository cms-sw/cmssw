import FWCore.ParameterSet.Config as cms

#AOD content
RecoJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_ak4CaloJets_*_*',
                                           'keep *_ak4CaloJets_rho_*',
                                           'keep *_ak4CaloJets_sigma_*',
                                           'keep *_ak4PFJetsCHS_*_*',
                                           'keep floatedmValueMap_puppi_*_*',
                                           'keep *_ak4PFJetsPuppi_*_*',
                                           'keep *_ak8PFJetsPuppi_*_*',
                                           'keep *_ak8PFJetsPuppiSoftDrop_*_*',
                                           'keep recoPFJets_ak4PFJets_*_*',
                                           'keep *_ak4PFJets_rho_*',
                                           'keep *_ak4PFJets_sigma_*',
                                           'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*',    
                                           'keep *_caloTowers_*_*', 
                                           'keep *_CastorTowerReco_*_*',                                           
                                           'keep *_ak4JetTracksAssociatorAtVertex_*_*',
                                           'keep *_ak4JetTracksAssociatorAtVertexPF_*_*', 
                                           'keep *_ak4JetTracksAssociatorExplicit_*_*',
                                           'keep *_ak4JetExtender_*_*', 
                                           'keep *_ak4JetID_*_*',
                                           'keep recoBasicJets_ak5CastorJets_*_*',
                                           'keep *_ak5CastorJets_rho_*',
                                           'keep *_ak5CastorJets_sigma_*',
                                           'keep *_ak5CastorJetID_*_*',
                                           'keep recoBasicJets_ak7CastorJets_*_*',
                                           'keep *_ak7CastorJets_rho_*',
                                           'keep *_ak7CastorJets_sigma_*',
                                           'keep *_ak7CastorJetID_*_*',
                                           'keep *_fixedGridRhoAll_*_*',
                                           'keep *_fixedGridRhoFastjetAll_*_*',
                                           'keep *_fixedGridRhoFastjetAllTmp_*_*',
                                           'keep *_fixedGridRhoFastjetCentral_*_*',
                                           'keep *_fixedGridRhoFastjetAllCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralCalo_*_*',
                                           'keep *_fixedGridRhoFastjetCentralChargedPileUp_*_*',
                                           'keep *_fixedGridRhoFastjetCentralNeutral_*_*',
                                           'keep *_ak8PFJetsPuppiSoftDropMass_*_*'
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
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
#products from regular pp which does not fit the normal AOD
for e in [pA_2016, peripheralPbPb, pp_on_XeXe_2017, pp_on_AA]:
    e.toModify( RecoJetsAOD.outputCommands, 
                func=lambda outputCommands: outputCommands.extend(['keep *_towerMaker_*_*'])
                )
for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toModify( RecoJetsAOD.outputCommands,
                func=lambda outputCommands: outputCommands.extend(['keep recoCentrality*_hiCentrality_*_*',
                                                                   'keep recoClusterCompatibility*_hiClusterCompatibility_*_*'
                                                                   ])
                )
#HI-specific products: needed in AOD, propagate to more inclusive tiers as well
pA_2016.toModify( RecoJetsAOD.outputCommands, 
                  func=lambda outputCommands: outputCommands.extend(['keep recoCentrality*_pACentrality_*_*',
                                                                     'keep *_hiFJGridEmptyAreaCalculator_*_*',
                                                                     'keep *_hiFJRhoProducer_*_*'
                                                                     ])
                )
#HI-specific products: needed in AOD, propagate to more inclusive tiers as well
peripheralPbPb.toModify( RecoJetsAOD.outputCommands, 
                         func=lambda outputCommands: outputCommands.extend(['keep recoCentrality*_pACentrality_*_*'])
                         )

pp_on_AA.toModify( RecoJetsAOD.outputCommands, 
                        func=lambda outputCommands: outputCommands.extend(['keep *_hiCentrality_*_*',
                                                                           'keep *_hiFJRhoProducer_*_*',
                                                                           'keep *_akPu3PFJets_*_*',
                                                                           'keep *_akPu4PFJets_*_*',
                                                                           'keep *_kt4PFJetsForRho_*_*',
                                                                           'keep *_akCs4PFJets_*_*',
                                                                           'keep *_akPu4CaloJets_*_*',
                                                                           'drop *_caloTowers_*_*'
                                                                           ])
                        )
#RECO content
RecoJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ak4CaloJets_*_*',
                                           'keep *_ak4PFJets_*_*',
                                           'keep *_ak4TrackJets_*_*',
                                           'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',         
                                           'keep *_towerMaker_*_*',
                                           'keep *_ak4JetTracksAssociatorAtCaloFace_*_*',
					   'keep *_ak5CastorJets_*_*',
                                           'keep *_ak7CastorJets_*_*',
                                           )
)
RecoJetsRECO.outputCommands.extend(RecoJetsAOD.outputCommands)

RecoGenJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoGenJetsRECO.outputCommands.extend(RecoGenJetsAOD.outputCommands)
#Full Event content 
RecoJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*_*_*', 
                                           'keep recoPFJets_*_*_*',
                                           'keep recoTrackJets_*_*_*',
                                           'keep recoJPTJets_*_*_*',
                                           'keep recoBasicJets_*_*_*',
                                           'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
                                           'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
                                           'keep *_kt4JetExtender_*_*',
                                           'keep *_ak7JetTracksAssociatorAtVertex*_*_*', 
                                           'keep *_ak7JetTracksAssociatorAtCaloFace*_*_*', 
                                           'keep *_ak7JetExtender_*_*',
                                           #keep jet area variables for jet colls in RECO 
                                           'keep *_kt4CaloJets_*_*', 
                                           'keep *_kt6CaloJets_*_*',
                                           'keep *_ak5CaloJets_*_*',
                                           'keep *_ak7CaloJets_*_*',
                                           'keep *_kt4PFJets_*_*', 
                                           'keep *_kt6PFJets_*_*',
                                           'keep *_ak5PFJets_*_*',
                                           'keep *_ak7PFJets_*_*',
                                           'keep *_kt4TrackJets_*_*',
                                           'keep *_ca*Mass_*_*',
                                           'keep *_ak*Mass_*_*'
        )
)
RecoJetsFEVT.outputCommands.extend(RecoJetsRECO.outputCommands)

RecoGenJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenJets_ak*_*_*')
)
RecoGenJetsFEVT.outputCommands.extend(RecoGenJetsRECO.outputCommands)
