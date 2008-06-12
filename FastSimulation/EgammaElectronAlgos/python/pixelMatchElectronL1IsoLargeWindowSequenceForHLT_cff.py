import FWCore.ParameterSet.Config as cms

import FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi

from FastSimulation.Configuration.blockHLT_cff import *

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
hltL1IsoLargeWindowElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1IsoLargeWindowTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# CTF track fit with material
ctfL1IsoLargeWindowTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfL1IsoLargeWindowWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("ctfL1IsoLargeWindowTracks"), cms.InputTag("hltCtfL1IsoWithMaterialTracks"))
)

# Electron collection
# include "RecoEgamma/EgammaHLTProducers/data/pixelMatchElectronsL1IsoLargeWindowForHLT.cff"
# sequence HLTPixelMatchElectronL1IsoLargeWindowSequence = {
#     hltL1IsoLargeWindowElectronPixelSeeds
# }
HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence = cms.Sequence(hltCkfL1IsoLargeWindowTrackCandidates+ctfL1IsoLargeWindowTracks+hltCtfL1IsoLargeWindowWithMaterialTracks+cms.SequencePlaceholder("hltPixelMatchElectronsL1IsoLargeWindow"))
# include "RecoEgamma/EgammaHLTProducers/data/pixelSeedConfigurationsForHLT.cfi"
hltL1IsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    # using l1IsoLargeWindowElectronPixelSeedConfiguration
    block_hltL1IsoLargeWindowElectronPixelSeeds
)
hltL1IsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
hltL1IsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated'
hltCkfL1IsoLargeWindowTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds")
hltCkfL1IsoLargeWindowTrackCandidates.TrackProducers = cms.VInputTag(cms.InputTag("hltCtfL1IsoWithMaterialTracks"))
hltCkfL1IsoLargeWindowTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1IsoLargeWindowTrackCandidates.SeedCleaning = True
ctfL1IsoLargeWindowTracks.src = 'hltCkfL1IsoLargeWindowTrackCandidates'
ctfL1IsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1IsoLargeWindowTracks.Fitter = 'KFFittingSmoother'
ctfL1IsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'

