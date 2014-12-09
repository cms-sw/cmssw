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

# (Not-so) Regional Tracking
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#hltL1NonIsoElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1NonIsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    # using l1NonIsoElectronSeedConfiguration
#    block_hltL1NonIsoElectronPixelSeeds
#)
#hltL1NonIsoElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
#hltL1NonIsoElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated'

#hltL1NonIsoStartUpElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1NonIsoStartUpElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1NonIsoStartUpElectronPixelSeeds
#)
#hltL1NonIsoStartUpElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
#hltL1NonIsoStartUpElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated'


# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi


hltCkfL1NonIsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1NonIsoTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoElectronPixelSeeds")
hltCkfL1NonIsoTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds")
#hltCkfL1NonIsoTrackCandidates.TrackProducers = []
hltCkfL1NonIsoTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfL1NonIsoTrackCandidates.SeedCleaning = True
hltCkfL1NonIsoTrackCandidates.SplitHits = False

#not needed 
#hltCkfL1NonIsoStartUpTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1NonIsoStartUpTrackCandidates.SeedProducer = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds")
#hltCkfL1NonIsoStartUpTrackCandidates.TrackProducers = []
#hltCkfL1NonIsoStartUpTrackCandidates.MaxNumberOfCrossedLayers = 999
#hltCkfL1NonIsoStartUpTrackCandidates.SeedCleaning = True
#hltCkfL1NonIsoStartUpTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtfL1NonIsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfL1NonIsoWithMaterialTracks.src = 'hltCkfL1NonIsoTrackCandidates'
hltCtfL1NonIsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1NonIsoWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtfL1NonIsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

#not needed
#hltCtfL1NonIsoStartUpWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
#hltCtfL1NonIsoStartUpWithMaterialTracks.src = 'hltCkfL1NonIsoStartUpTrackCandidates'
#hltCtfL1NonIsoStartUpWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
#hltCtfL1NonIsoStartUpWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
#hltCtfL1NonIsoStartUpWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

#Sequence
HLTPixelMatchElectronL1NonIsoTrackingSequence = cms.Sequence(globalPixelTracking +
                                                             hltCkfL1NonIsoTrackCandidates+
                                                             hltCtfL1NonIsoWithMaterialTracks+
                                                             cms.SequencePlaceholder("hltPixelMatchElectronsL1NonIso"))

#for debugging
#from FWCore.Modules.printContent_cfi import *

hltL1NonIsoStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking +
#                                                            printContent+
                                                            cms.SequencePlaceholder("hltL1NonIsoStartUpElectronPixelSeeds"))
                                                    
