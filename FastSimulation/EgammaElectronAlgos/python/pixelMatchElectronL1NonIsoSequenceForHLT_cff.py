import FWCore.ParameterSet.Config as cms

import FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi
#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons
# include "RecoEgamma/EgammaHLTProducers/data/egammaHLTChi2MeasurementEstimatorESProducer.cff"
# Cluster-seeded pixel pairs
hltL1NonIsoElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi
hltL1NonIsoStartUpElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# include "RecoEgamma/EgammaHLTProducers/data/pixelMatchElectronsL1NonIsoForHLT.cff"
# CKFTrackCandidateMaker
hltCkfL1NonIsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1NonIsoStartUpTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# CTF track fit with material
hltCtfL1NonIsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltCtfL1NonIsoStartUpWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
# include "RecoEgamma/EgammaHLTProducers/data/pixelSeedConfigurationsForHLT.cfi"
hltL1NonIsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
    # using l1NonIsoElectronPixelSeedConfiguration
    block_hltL1NonIsoElectronPixelSeeds
)
hltL1NonIsoElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
hltL1NonIsoElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated'
hltL1NonIsoStartUpElectronPixelSeeds.SeedConfiguration = cms.PSet(
    block_hltL1NonIsoStartUpElectronPixelSeeds
)
hltL1NonIsoStartUpElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
hltL1NonIsoStartUpElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated'
hltCkfL1NonIsoTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoElectronPixelSeeds")
hltCkfL1NonIsoTrackCandidates.TrackProducers = []
hltCkfL1NonIsoTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1NonIsoTrackCandidates.SeedCleaning = True
hltCkfL1NonIsoStartUpTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds")
hltCkfL1NonIsoStartUpTrackCandidates.TrackProducers = []
hltCkfL1NonIsoStartUpTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1NonIsoStartUpTrackCandidates.SeedCleaning = True
hltCtfL1NonIsoWithMaterialTracks.src = 'hltCkfL1NonIsoTrackCandidates'
hltCtfL1NonIsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1NonIsoWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltCtfL1NonIsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'
hltCtfL1NonIsoStartUpWithMaterialTracks.src = 'hltCkfL1NonIsoStartUpTrackCandidates'
hltCtfL1NonIsoStartUpWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1NonIsoStartUpWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltCtfL1NonIsoStartUpWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

