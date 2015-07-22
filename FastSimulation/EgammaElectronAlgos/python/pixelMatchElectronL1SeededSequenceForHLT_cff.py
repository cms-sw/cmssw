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


hltCkfL1SeededTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1SeededTrackCandidates.src = cms.InputTag("hltL1SeededElectronPixelSeeds")
hltCkfL1SeededTrackCandidates.src = cms.InputTag("hltL1SeededStartUpElectronPixelSeeds")
#hltCkfL1SeededTrackCandidates.TrackProducers = []
hltCkfL1SeededTrackCandidates.SplitHits = False

hltL1SeededCkfTrackCandidatesForGSF = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltL1SeededCkfTrackCandidatesForGSF.src = cms.InputTag("hltL1SeededStartUpElectronPixelSeeds")
#hltL1SeededCkfTrackCandidatesForGSF.TrackProducers = []
hltL1SeededCkfTrackCandidatesForGSF.SplitHits = True

#not needed 
#hltCkfL1NonIsoStartUpTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
#hltCkfL1NonIsoStartUpTrackCandidates.src = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds")
#hltCkfL1NonIsoStartUpTrackCandidates.TrackProducers = []
#hltCkfL1NonIsoStartUpTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtfL1SeededWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfL1SeededWithMaterialTracks.src = 'hltCkfL1SeededTrackCandidates'
hltCtfL1SeededWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfL1SeededWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtfL1SeededWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

#not needed
#hltCtfL1NonIsoStartUpWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
#hltCtfL1NonIsoStartUpWithMaterialTracks.src = 'hltCkfL1NonIsoStartUpTrackCandidates'
#hltCtfL1NonIsoStartUpWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
#hltCtfL1NonIsoStartUpWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
#hltCtfL1NonIsoStartUpWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

#Sequence
HLTPixelMatchElectronL1SeededTrackingSequence = cms.Sequence(globalPixelTracking +
                                                             hltCkfL1SeededTrackCandidates+
                                                             hltCtfL1SeededWithMaterialTracks+
                                                             cms.SequencePlaceholder("hltPixelMatchElectronsL1Seeded"))

#for debugging
#from FWCore.Modules.printContent_cfi import *

hltL1SeededStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking +
#                                                            printContent+
                                                            cms.SequencePlaceholder("hltL1SeededStartUpElectronPixelSeeds"))
                                                    
