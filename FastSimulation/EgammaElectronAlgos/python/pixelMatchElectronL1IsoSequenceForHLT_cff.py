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
hltL1IsoElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi
hltL1IsoStartUpElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# CKFTrackCandidateMaker
hltCkfL1IsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1IsoStartUpTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# CTF track fit with material
hltCtfL1IsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltCtfL1IsoStartUpWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
# include "RecoEgamma/EgammaHLTProducers/data/pixelSeedConfigurationsForHLT.cfi"
hltL1IsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
    # using l1IsoElectronPixelSeedConfiguration
    block_hltL1IsoElectronPixelSeeds
)
hltL1IsoElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
hltL1IsoElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated'
hltL1IsoStartUpElectronPixelSeeds.SeedConfiguration = cms.PSet(
    # using l1IsoElectronPixelSeedConfiguration
    block_hltL1IsoStartUpElectronPixelSeeds
)
hltL1IsoStartUpElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
hltL1IsoStartUpElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated'
hltCkfL1IsoTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoElectronPixelSeeds")
hltCkfL1IsoTrackCandidates.TrackProducers = []
hltCkfL1IsoTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1IsoTrackCandidates.SeedCleaning = True
hltCkfL1IsoStartUpTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds")
hltCkfL1IsoStartUpTrackCandidates.TrackProducers = []
hltCkfL1IsoStartUpTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1IsoStartUpTrackCandidates.SeedCleaning = True
hltCtfL1IsoWithMaterialTracks.src = 'hltCkfL1IsoTrackCandidates'
hltCtfL1IsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1IsoWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltCtfL1IsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'
hltCtfL1IsoStartUpWithMaterialTracks.src = 'hltCkfL1IsoStartUpTrackCandidates'
hltCtfL1IsoStartUpWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1IsoStartUpWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltCtfL1IsoStartUpWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

