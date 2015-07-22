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
hltCkf3HitL1SeededTrackCandidates.src = cms.InputTag("hltL1SeededStartUpElectronPixelSeeds")
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

#### ISO sequence

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkf3HitL1IsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkf3HitL1IsoTrackCandidates.src = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds")
hltCkf3HitL1IsoTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtf3HitL1IsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtf3HitL1IsoWithMaterialTracks.src = 'hltCkfL1IsoTrackCandidates'
hltCtf3HitL1IsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtf3HitL1IsoWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtf3HitL1IsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'


#hltL1IsoStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
#                                                         cms.SequencePlaceholder("hltL1IsoStartUpElectronPixelSeeds"))

HLTPixelMatch3HitElectronL1IsoTrackingSequence = cms.Sequence(hltCkf3HitL1IsoTrackCandidates+
                                                              hltCtf3HitL1IsoWithMaterialTracks+
                                                              cms.SequencePlaceholder("hltPixelMatch3HitElectronsL1Iso"))


#### NON-ISO sequence

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkf3HitL1NonIsoTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkf3HitL1NonIsoTrackCandidates.src = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds")
hltCkf3HitL1NonIsoTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtf3HitL1NonIsoWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtf3HitL1NonIsoWithMaterialTracks.src = 'hltCkfL1NonIsoTrackCandidates'
hltCtf3HitL1NonIsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtf3HitL1NonIsoWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtf3HitL1NonIsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'


#hltL1NonIsoStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
#                                                         cms.SequencePlaceholder("hltL1NonIsoStartUpElectronPixelSeeds"))

HLTPixelMatch3HitElectronL1IsoTrackingSequence = cms.Sequence(hltCkf3HitL1NonIsoTrackCandidates+
                                                              hltCtf3HitL1NonIsoWithMaterialTracks+
                                                              cms.SequencePlaceholder("hltPixelMatch3HitElectronsL1NonIso"))


#### Activity sequence

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkf3HitActivityTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkf3HitActivityTrackCandidates.src = cms.InputTag("hltActivityStartUpElectronPixelSeeds")
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

