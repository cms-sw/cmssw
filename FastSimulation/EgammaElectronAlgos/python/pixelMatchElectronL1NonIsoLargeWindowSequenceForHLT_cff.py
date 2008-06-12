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
hltL1NonIsoLargeWindowElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi.electronGSPixelSeeds.clone()
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1NonIsoLargeWindowTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# CTF track fit with material
ctfL1NonIsoLargeWindowTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfL1NonIsoLargeWindowWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("ctfL1NonIsoLargeWindowTracks"), cms.InputTag("hltCtfL1NonIsoWithMaterialTracks"))
)

# Electron collection
# include "RecoEgamma/EgammaHLTProducers/data/pixelMatchElectronsL1NonIsoLargeWindowForHLT.cff"
# sequence HLTPixelMatchElectronL1NonIsoLargeWindowSequence = {
#     hltL1NonIsoLargeWindowElectronPixelSeeds
# }
HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence = cms.Sequence(hltCkfL1NonIsoLargeWindowTrackCandidates+ctfL1NonIsoLargeWindowTracks+hltCtfL1NonIsoLargeWindowWithMaterialTracks+cms.SequencePlaceholder("hltPixelMatchElectronsL1NonIsoLargeWindow"))
hltL1NonIsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    block_hltL1NonIsoLargeWindowElectronPixelSeeds
)
hltL1NonIsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
hltL1NonIsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated'
hltCkfL1NonIsoLargeWindowTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds")
hltCkfL1NonIsoLargeWindowTrackCandidates.TrackProducers = cms.VInputTag(cms.InputTag("hltCtfL1NonIsoWithMaterialTracks"))
hltCkfL1NonIsoLargeWindowTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1NonIsoLargeWindowTrackCandidates.SeedCleaning = True
ctfL1NonIsoLargeWindowTracks.src = 'hltCkfL1NonIsoLargeWindowTrackCandidates'
ctfL1NonIsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1NonIsoLargeWindowTracks.Fitter = 'KFFittingSmoother'
ctfL1NonIsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'

