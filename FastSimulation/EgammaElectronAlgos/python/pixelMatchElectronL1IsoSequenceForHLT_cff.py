import FWCore.ParameterSet.Config as cms
#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons

# Cluster-seeded pixel pairs
#import FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi
#from FastSimulation.Configuration.blockHLT_8E29_cff import *

# (Not-so) Regional Tracking - needed because the ElectronSeedProducer needs the seeds 
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#hltL1IsoElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1IsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1IsoElectronPixelSeeds
#)
#hltL1IsoElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
#hltL1IsoElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated'

#hltL1IsoStartUpElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1IsoStartUpElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1IsoStartUpElectronPixelSeeds
#)
#hltL1IsoStartUpElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
#hltL1IsoStartUpElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated'

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkfL1IsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1IsoTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoElectronPixelSeeds")
hltCkfL1IsoTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds")
hltCkfL1IsoTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1IsoTrackCandidates.SeedCleaning = True
hltCkfL1IsoTrackCandidates.SplitHits = False

# Not needed any longer 
#hltCkfL1IsoStartUpTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1IsoStartUpTrackCandidates.SeedProducer = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds")
#hltCkfL1IsoStartUpTrackCandidates.TrackProducers = []
#hltCkfL1IsoStartUpTrackCandidates.MaxNumberOfCrossedLayers = 999
#hltCkfL1IsoStartUpTrackCandidates.SeedCleaning = True
#hltCkfL1IsoStartUpTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtfL1IsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfL1IsoWithMaterialTracks.src = 'hltCkfL1IsoTrackCandidates'
hltCtfL1IsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1IsoWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtfL1IsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

# not needed 
#hltCtfL1IsoStartUpWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
#hltCtfL1IsoStartUpWithMaterialTracks.src = 'hltCkfL1IsoStartUpTrackCandidates'
#hltCtfL1IsoStartUpWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
#hltCtfL1IsoStartUpWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
#hltCtfL1IsoStartUpWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

#for debugging
#from FWCore.Modules.printContent_cfi import *
hltL1IsoStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
#                                                         printContent+
                                                         cms.SequencePlaceholder("hltL1IsoStartUpElectronPixelSeeds"))

HLTPixelMatchElectronL1IsoTrackingSequence = cms.Sequence(hltCkfL1IsoTrackCandidates+
                                                          hltCtfL1IsoWithMaterialTracks+
                                                          cms.SequencePlaceholder("hltPixelMatchElectronsL1Iso"))
