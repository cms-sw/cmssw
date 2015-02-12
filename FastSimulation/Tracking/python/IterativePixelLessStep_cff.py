import FWCore.ParameterSet.Config as cms

# step 4

# seeding
#from FastSimulation.Tracking.IterativeFourthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelLessSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepIds"), 
    cms.InputTag("detachedTripletStepIds"), 
    cms.InputTag("lowPtTripletStepIds"), 
    cms.InputTag("pixelPairStepIds"),  
    cms.InputTag("mixedTripletStepIds")
    ]
iterativePixelLessSeeds.simTrackSelection.pTMin = 0.3
iterativePixelLessSeeds.simTrackSelection.maxD0 = 99.
iterativePixelLessSeeds.simTrackSelection.maxZ0 = 99.
iterativePixelLessSeeds.minLayersCrossed = 3
iterativePixelLessSeeds.originRadius = 1.0
iterativePixelLessSeeds.originHalfLength = 12.0
iterativePixelLessSeeds.originpTMin = 0.4 # was 0.6

iterativePixelLessSeeds.primaryVertex = ''

#iterativePixelLessSeeds.layerList = ['TIB1+TIB2',
#                                     'TIB1+TID1_pos','TIB1+TID1_neg',
#                                     'TID3_pos+TEC1_pos','TID3_neg+TEC1_neg',
#                                     'TID1_pos+TID2_pos','TID2_pos+TID3_pos',
#                                     'TEC1_pos+TEC2_pos','TEC2_pos+TEC3_pos','TEC3_pos+TEC4_pos','TEC3_pos+TEC5_pos','TEC4_pos+TEC5_pos',
#                                     'TID1_neg+TID2_neg','TID2_neg+TID3_neg',
#                                     'TEC1_neg+TEC2_neg','TEC2_neg+TEC3_neg','TEC3_neg+TEC4_neg','TEC3_neg+TEC5_neg','TEC4_neg+TEC5_neg']
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepSeedLayers
iterativePixelLessSeeds.layerList = pixelLessStepSeedLayers.layerList

# candidate producer
#from FastSimulation.Tracking.IterativeFourthCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelLessTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelLessTrackCandidates.SeedProducer = cms.InputTag("iterativePixelLessSeeds")
iterativePixelLessTrackCandidates.MinNumberOfCrossedLayers = 6 # was 5


# track producer
#from FastSimulation.Tracking.IterativeFourthTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelLessTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelLessTracks.src = 'iterativePixelLessTrackCandidates'
iterativePixelLessTracks.TTRHBuilder = 'WithoutRefit'
##iterativePixelLessTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativePixelLessTracks.Fitter = 'KFFittingSmootherFourth'
iterativePixelLessTracks.Propagator = 'PropagatorWithMaterial'
iterativePixelLessTracks.AlgorithmName = cms.string('pixelLessStep')

# simtrack id producer
pixelLessStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativePixelLessTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )






# track selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelLessStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativePixelLessTracks',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'pixelLessStepLoose',
                            chi2n_par = 0.5,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1.5, 4.0 ),
                            dz_par1 = ( 1.5, 4.0 ),
                            d0_par2 = ( 1.5, 4.0 ),
                            dz_par2 = ( 1.5, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'pixelLessStepTight',
                            preFilterName = 'pixelLessStepLoose',
                            chi2n_par = 0.35,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 0,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1.2, 4.0 ),
                            dz_par1 = ( 1.2, 4.0 ),
                            d0_par2 = ( 1.2, 4.0 ),
                            dz_par2 = ( 1.2, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'pixelLessStep',
                            preFilterName = 'pixelLessStepTight',
                            chi2n_par = 0.25,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 0,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1., 4.0 ),
                            dz_par1 = ( 1., 4.0 ),
                            d0_par2 = ( 1., 4.0 ),
                            dz_par2 = ( 1., 4.0 )
                            ),
                    ) #end of vpset
            ) #end of clone


# sequence
iterativePixelLessStep = cms.Sequence(iterativePixelLessSeeds+
                                      iterativePixelLessTrackCandidates+
                                      iterativePixelLessTracks+
                                      pixelLessStepIds+
                                      pixelLessStepSelector)
