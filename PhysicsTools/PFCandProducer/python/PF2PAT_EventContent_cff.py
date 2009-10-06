
import FWCore.ParameterSet.Config as cms


PF2PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # Gen information
    'keep *_genParticles_*_*',
    'keep *_genMetTrue_*_*',
    'keep recoGenJets_*_*_*',
    # isolated electrons and muons
    'keep *_pfIsolatedElectrons_*_*',
    'keep *_pfIsolatedMuons_*_*',
    'keep *_pfNoJet_*_*',
    'keep recoIsoDepositedmValueMap_*_*_*',
    # jets
    'keep recoPFJets_pfNoTau_*_*',
    # taus 
    'keep *_allLayer0Taus_*_*',
    'keep recoPFTauDiscriminator_*_*_*',
    'keep *_offlinePrimaryVerticesWithBS_*_*',
    # MET
    'keep *_pfMET_*_*'
    )
)

PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # Gen information
#    'keep *_genParticles_*_*',
    'keep *_genMetTrue_*_*',
    'keep recoGenJets_iterativeCone5GenJets_*_*',
    'keep patElectrons_*_*_*',
    'keep patMuons_*_*_*',
    'keep patJets_*_*_*',
    'keep patMET_*_*_*',
    'keep patTaus_*_*_*',
    'keep recoIsoDepositedmValueMap_iso*_*_*'
    )
)

PF2PATStudiesEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPFJets_*_*_*',
    'keep *_decaysFromZs_*_*',
    'keep recoPFCandidates_*_*_PF2PAT',
    'keep recoPFCandidates_*_*_PAT',    
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoMuons_*_*_*',
    'keep *_pf*_*_*'
    )
)

prunedAODForPF2PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop recoCaloTau*_*_*_*',
        'drop recoPFTau*_*_*_*',
        'drop recoCaloJet*_*_*_*',
        'drop recoPFJet*_*_*_*',
        'drop recoConversions_*_*_*', 
        'drop recoJetedmRefToBaseProdTofloatsAssociationVector_*_*_*',
        'drop recoPreshowerClusters_*_*_*',
        'drop recoMETs_*_*_*',
        'drop recoPFMETs_*_*_*',
        'drop recoCaloMETs_*_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_tevMuons_*_*',
        'drop *_generalV0Candidates_*_*',
        'drop *_*TracksFromConversions_*_*',
        'drop recoPhoton*_*_*_*',
        'drop *_muIsoDeposit*_*_*',
        'drop recoMuonMETCorrectionDataedmValueMap_*_*_*',
        'drop *_*JetTracksAssociator*_*_*',
        'drop *_*JetExtender_*_*',
        'drop recoSoftLeptonTagInfos_*_*_*',
        'drop *_impactParameterTagInfos_*_*',
        'drop *_towerMaker_*_*',
        'drop *_sisCone*_*_*',
        'drop *_PhotonIDProd_*_*',
        'drop recoHFEMClusterShapes_*_*_*', 
        'drop recoCaloClustersToOnereco*_*_*_*',
        'drop EcalRecHitsSorted_*_*_*',
        'drop recoCaloClusters_*_*_*',
# needed somewhere in PAT. and could be useful in the future. 
#        'drop *_softPFElectrons_*_*',
        'drop *_particleFlow_electrons_*',
        'drop recoPreshowerClusterShapes_*_*_*',
# needed in PAT by allLayer1Electrons - dunno why:
#        'drop *_gsfElectronCores_*_*',
        'drop *_hfRecoEcalCandidate_*_*',
        'drop recoSuperClusters_*_*_*',
        'keep *_pfElectronTranslator_*_*',
        'keep recoSuperClusters_corrected*_*_*'
        )
)
