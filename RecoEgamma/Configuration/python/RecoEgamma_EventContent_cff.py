import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_gsfElectronCores_*_*', 
        'keep *_gsfElectrons_*_*', 
        'keep *_uncleanedOnlyGsfElectronCores_*_*', 
        'keep *_uncleanedOnlyGsfElectrons_*_*', 
        'keep *_eidRobustLoose_*_*',
        'keep *_eidRobustTight_*_*',
        'keep *_eidRobustHighEnergy_*_*',
        'keep *_eidLoose_*_*',
        'keep *_eidTight_*_*',
        'keep *_egmGedGsfElectronPF*Isolation_*_*',
        'keep *_egmGsfElectronIDs_*_*', 
        'keep *_egmPhotonIDs_*_*',
        'keep *_conversions_*_*',
        'keep *_mustacheConversions_*_*',
        'drop *_conversions_uncleanedConversions_*',
        'keep *_gedPhotonCore_*_*',
        'keep *_gedPhotonsTmp_*_*',
        'keep *_gedPhotons_*_*',
        'keep *_particleBasedIsolation_*_*',
        'keep *_photonCore_*_*',
        'keep *_photons_*_*',
        'keep *_mustachePhotonCore_*_*',
        'keep *_mustachePhotons_*_*',
        'keep *_ootPhotonCore_*_*',
        'keep *_ootPhotons_*_*',
        'keep *_allConversions_*_*',
        'keep *_allConversionsOldEG_*_*',
        'keep *_ckfOutInTracksFrom*Conversions_*_*', 
        'keep *_ckfInOutTracksFrom*Conversions_*_*',
        'keep *_uncleanedOnlyAllConversions_*_*',
        'keep *_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 
        'keep *_uncleanedOnlyCkfInOutTracksFromConversions_*_*',
        'keep *_PhotonIDProd_*_*',
        'keep *_PhotonIDProdGED_*_*',
        'keep *_hfRecoEcalCandidate_*_*',
        'keep *_hfEMClusters_*_*',
        'keep *_gedGsfElectronCores_*_*',
        'keep *_gedGsfElectrons_*_*'
  )                                                                 
)

# RECO content
RecoEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoGsfElectronCores_gsfElectronCores_*_*',
        'keep recoGsfElectronCores_gedGsfElectronCores_*_*',
        'keep recoGsfElectrons_gsfElectrons_*_*',
        'keep recoGsfElectrons_gedGsfElectrons_*_*',
        'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*', 
        'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*', 
        'keep floatedmValueMap_eidRobustLoose_*_*',
        'keep floatedmValueMap_eidRobustTight_*_*',
        'keep floatedmValueMap_eidRobustHighEnergy_*_*',
        'keep floatedmValueMap_eidLoose_*_*',
        'keep floatedmValueMap_eidTight_*_*',
        'keep *_egmGedGsfElectronPFIsolation_*_*',
        'drop *_egmGsfElectronIDs_*_*',
        'drop *_egmPhotonIDs_*_*',
        'keep *_gedPhotonCore_*_*',
        'keep *_gedPhotons_*_*',
        'keep *_particleBasedIsolation_*_*',
        #'keep *_gedPhotonsTmp_*_*',        
        'keep recoPhotons_mustachePhotons_*_*',
        'keep recoPhotonCores_mustachePhotonCore_*_*',
        'keep recoPhotons_ootPhotons_*_*',
        'keep recoPhotonCores_ootPhotonCore_*_*',
        'keep recoPhotons_photons_*_*',
        'keep recoPhotonCores_photonCore_*_*', 
        'keep recoConversions_conversions_*_*',
        'keep recoConversions_mustacheConversions_*_*', 
        'drop *_conversions_uncleanedConversions_*',
        'drop *_gedPhotonsTmp_valMapPFEgammaCandToPhoton_*',
        'keep recoConversions_allConversions_*_*',
        'keep recoConversions_allConversionsOldEG_*_*',
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*', 
        'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 
        'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*',
        'keep recoConversions_uncleanedOnlyAllConversions_*_*',
        'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 
        'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 
        'keep recoTrackExtras_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 
        'keep recoTrackExtras_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_uncleanedOnlyCkfInOutTracksFromConversions_*_*',
        'keep *_PhotonIDProd_*_*',
        'keep *_PhotonIDProdGED_*_*',
        #'keep *_hfRecoEcalCandidate_*_*',
        'keep recoRecoEcalCandidates_hfRecoEcalCandidate_*_*',
        'keep *_hfEMClusters_*_*',
        'keep *_gedGsfElectronCores_*_*',
        'keep *_gedGsfElectrons_*_*',
        'keep *_lowPtGsfEle*_*_*',
  )                                                                 
)

# AOD content
RecoEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoGsfElectronCores_gsfElectronCores_*_*', 
        'keep recoGsfElectronCores_gedGsfElectronCores_*_*',
        'keep recoGsfElectrons_gsfElectrons_*_*', 
        'keep recoGsfElectrons_gedGsfElectrons_*_*', 
        'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*', 
        'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*', 
        'keep floatedmValueMap_eidRobustLoose_*_*',
        'keep floatedmValueMap_eidRobustTight_*_*',
        'keep floatedmValueMap_eidRobustHighEnergy_*_*',
        'keep floatedmValueMap_eidLoose_*_*',
        'keep floatedmValueMap_eidTight_*_*',
        'keep *_egmGedGsfElectronPFIsolation_*_*',
        'drop *_egmGsfElectronIDs_*_*',
        'drop *_egmPhotonIDs_*_*',
        'keep recoPhotonCores_gedPhotonCore_*_*',
        'keep recoPhotons_gedPhotons_*_*',
        'keep *_particleBasedIsolation_*_*',
        'drop *_gedPhotons_valMapPFEgammaCandToPhoton_*',
        'keep recoPhotonCores_photonCore_*_*',
        'keep recoPhotons_photons_*_*', 
        'keep recoPhotonCores_ootPhotonCore_*_*',
        'keep recoPhotons_ootPhotons_*_*',
        'keep recoConversions_conversions_*_*',
        'keep recoConversions_mustacheConversions_*_*',
        'drop *_conversions_uncleanedConversions_*',
        'keep recoConversions_allConversions_*_*',
        'keep recoConversions_allConversionsOldEG_*_*',
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*',
        'keep recoConversions_uncleanedOnlyAllConversions_*_*',
        'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 
        'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*',
        'keep *_PhotonIDProd_*_*',
        'keep *_PhotonIDProdGED_*_*',
        'keep *_hfRecoEcalCandidate_*_*',
        'keep recoRecoEcalCandidates_hfRecoEcalCandidate_*_*',
        'keep *_hfEMClusters_*_*',
        'keep *_gedGsfElectronCores_*_*',
        'keep *_gedGsfElectrons_*_*',
        'keep *_lowPtGsfEle*_*_*',
  )                                                                 
)

# mods for HGCAL
_phase2_hgcal_RecoEgamma_tokeep = [ 'keep *_ecalDrivenGsfElectronCores_*_*',
                                    'keep *_ecalDrivenGsfElectrons_*_*',
                                    'keep *_ecalDrivenGsfElectronCoresFromMultiCl_*_*',
                                    'keep *_ecalDrivenGsfElectronsFromMultiCl_*_*',
                                    'keep *_photonCoreFromMultiCl_*_*',
                                    'keep *_photonsFromMultiCl_*_*']
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( RecoEgammaFEVT, outputCommands = RecoEgammaFEVT.outputCommands + _phase2_hgcal_RecoEgamma_tokeep
)
phase2_hgcal.toModify( RecoEgammaRECO, outputCommands = RecoEgammaRECO.outputCommands + _phase2_hgcal_RecoEgamma_tokeep )
phase2_hgcal.toModify( RecoEgammaAOD,  outputCommands = RecoEgammaAOD.outputCommands + _phase2_hgcal_RecoEgamma_tokeep )

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
#HI-specific products needed in pp scenario special configurations
for e in [pA_2016, peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017, ppRef_2017]:
    for ec in [RecoEgammaAOD.outputCommands, RecoEgammaRECO.outputCommands, RecoEgammaFEVT.outputCommands]:
        e.toModify( ec, func=lambda outputCommands: outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                                           'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                                           'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                                           'keep recoPhotonCores_islandPhotonCore_*_*',
                                                                           'keep recoPhotons_islandPhotons_*_*'
                                                                           ])
                    )
