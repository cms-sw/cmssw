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

# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkfActivityTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkfActivityTrackCandidates.SeedProducer = cms.InputTag("hltActivityStartUpElectronPixelSeeds")
hltCkfActivityTrackCandidates.MaxNumberOfCrossedLayers = 999
hltCkfActivityTrackCandidates.SeedCleaning = True
hltCkfActivityTrackCandidates.SplitHits = False

hltActivityCkfTrackCandidatesForGSF = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltActivityCkfTrackCandidatesForGSF.SeedProducer = cms.InputTag("hltActivityStartUpElectronPixelSeeds")
hltActivityCkfTrackCandidatesForGSF.MaxNumberOfCrossedLayers = 999
hltActivityCkfTrackCandidatesForGSF.SeedCleaning = True
hltActivityCkfTrackCandidatesForGSF.SplitHits = True



# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltCtfActivityWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfActivityWithMaterialTracks.src = 'hltCkfActivityTrackCandidates'
hltCtfActivityWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltCtfActivityWithMaterialTracks.Fitter = 'KFFittingSmootherForElectrons'
hltCtfActivityWithMaterialTracks.Propagator = 'PropagatorWithMaterial'


hltLActivityStartUpElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
                                                         cms.SequencePlaceholder("hltActivityStartUpElectronPixelSeeds"))

HLTPixelMatchElectronActivityTrackingSequence = cms.Sequence(hltCkfActivityTrackCandidates+
                                                          hltCtfActivityWithMaterialTracks+
                                                          cms.SequencePlaceholder("hltPixelMatchElectronsActivity"))


