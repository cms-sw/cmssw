import FWCore.ParameterSet.Config as cms
#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons

# (Not-so) Regional Tracking - needed because the ElectronSeedProducer needs the seeds 
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#### Modified 52X filter sequence

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkf3HitL1SeededTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkf3HitL1SeededTrackCandidates.SeedProducer = cms.InputTag("hltL1SeededStartUpElectronPixelSeeds")
hltCkf3HitL1SeededTrackCandidates.TrackProducers = []
hltCkf3HitL1SeededTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkf3HitL1SeededTrackCandidates.SeedCleaning = True
hltCkf3HitL1SeededTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtf3HitL1SeededWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtf3HitL1SeededWithMaterialTracks.src = 'hltCkf3HitL1SeededTrackCandidates'
hltCtf3HitL1SeededWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtf3HitL1SeededWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtf3HitL1SeededWithMaterialTracks.Propagator = 'PropagatorWithMaterial'


#hltL1SeededStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
#                                                         cms.SequencePlaceholder("hltL1SeededStartUpElectronPixelSeeds"))

HLTPixelMatch3HitElectronL1SeededTrackingSequence = cms.Sequence(hltCkf3HitL1SeededTrackCandidates+
                                                          hltCtf3HitL1SeededWithMaterialTracks+
                                                          cms.SequencePlaceholder("hltPixelMatch3HitElectronsL1Seeded"))


#### Activity sequence

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkf3HitActivityTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkf3HitActivityTrackCandidates.SeedProducer = cms.InputTag("hltActivityStartUpElectronPixelSeeds")
hltCkf3HitActivityTrackCandidates.TrackProducers = []
hltCkf3HitActivityTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkf3HitActivityTrackCandidates.SeedCleaning = True
hltCkf3HitActivityTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtf3HitActivityWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtf3HitActivityWithMaterialTracks.src = 'hltCkf3HitActivityTrackCandidates'
hltCtf3HitActivityWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtf3HitActivityWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtf3HitActivityWithMaterialTracks.Propagator = 'PropagatorWithMaterial'


#hltActivityStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
#                                                         cms.SequencePlaceholder("hltActivityStartUpElectronPixelSeeds"))

HLTPixelMatch3HitElectronActivityTrackingSequence = cms.Sequence(hltCkf3HitActivityTrackCandidates+
                                                                  hltCtf3HitActivityWithMaterialTracks+
                                                                  cms.SequencePlaceholder("hltPixelMatch3HitElectronsActivity"))

