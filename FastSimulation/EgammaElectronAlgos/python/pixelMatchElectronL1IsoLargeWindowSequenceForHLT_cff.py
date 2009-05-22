import FWCore.ParameterSet.Config as cms


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
import FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi
from FastSimulation.Configuration.blockHLT_8E29_cff import *

hltL1IsoLargeWindowElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
hltL1IsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    block_hltL1IsoLargeWindowElectronPixelSeeds
)
hltL1IsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
hltL1IsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated'

# Track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1IsoLargeWindowTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkfL1IsoLargeWindowTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds")
hltCkfL1IsoLargeWindowTrackCandidates.TrackProducers = cms.VInputTag(cms.InputTag("hltCtfL1IsoWithMaterialTracks"))
hltCkfL1IsoLargeWindowTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1IsoLargeWindowTrackCandidates.SeedCleaning = True
hltCkfL1IsoLargeWindowTrackCandidates.SplitHits = False


# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
ctfL1IsoLargeWindowTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
ctfL1IsoLargeWindowTracks.src = 'hltCkfL1IsoLargeWindowTrackCandidates'
ctfL1IsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1IsoLargeWindowTracks.Fitter = 'KFFittingSmootherForElectrons'
ctfL1IsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'


# Merge
hltCtfL1IsoLargeWindowWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("ctfL1IsoLargeWindowTracks"),
                                   cms.InputTag("hltCtfL1IsoWithMaterialTracks"))
)

# Sequence for electrons iso, large window.
#HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence = cms.Sequence(hltCkfL1IsoLargeWindowTrackCandidates+
#                                                                     ctfL1IsoLargeWindowTracks+
#                                                                     hltCtfL1IsoLargeWindowWithMaterialTracks+
#                                                                     cms.SequencePlaceholder("hltPixelMatchElectronsL1IsoLargeWindow"))


HLTPixelMatchLargeWindowElectronL1IsoTrackingSequence = cms.Sequence(hltCkfL1IsoLargeWindowTrackCandidates+
                                                                     ctfL1IsoLargeWindowTracks+
                                                                     hltCtfL1IsoLargeWindowWithMaterialTracks+
                                                                     cms.SequencePlaceholder("hltPixelMatchLargeWindowElectronsL1Iso"))

