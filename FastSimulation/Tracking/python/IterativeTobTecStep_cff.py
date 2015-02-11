import FWCore.ParameterSet.Config as cms

# step 5

# seeding
#from FastSimulation.Tracking.IterativeFifthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeTobTecSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeTobTecSeeds.simTrackSelection.skipSimTrackIds = [cms.InputTag("initialStepIds"), cms.InputTag("lowPtTripletStepIds"), cms.InputTag("pixelPairStepIds"), cms.InputTag("detachedTripletStepIds"), cms.InputTag("mixedTripletStepIds"), cms.InputTag("pixelLessStepIds")]
iterativeTobTecSeeds.simTrackSelection.pTMin = 0.3
iterativeTobTecSeeds.simTrackSelection.maxD0 = 99.
iterativeTobTecSeeds.simTrackSelection.maxZ0 = 99.
iterativeTobTecSeeds.minLayersCrossed = 4
iterativeTobTecSeeds.outputSeedCollectionName = 'TobTecLayerPairs'
iterativeTobTecSeeds.originRadius = 6.0 # was 5.0
iterativeTobTecSeeds.originHalfLength = 30.0 # was 10.0
iterativeTobTecSeeds.originpTMin = 0.6 # was 0.5

iterativeTobTecSeeds.primaryVertex = ''

#iterativeTobTecSeeds.layerList = ['TOB1+TOB2', 
#                                  'TOB1+TEC1_pos', 'TOB1+TEC1_neg', 
#                                  'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 
#                                  'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 
#                                  'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 
#                                  'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 
#                                  'TEC3_neg+TEC4_neg', 'TEC4_neg+TEC5_neg', 
#                                  'TEC5_neg+TEC6_neg', 'TEC6_neg+TEC7_neg']
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair
iterativeTobTecSeeds.layerList = ['TOB1+TOB2']
iterativeTobTecSeeds.layerList.extend(tobTecStepSeedLayersPair.layerList)

# candidate producer
#from FastSimulation.Tracking.IterativeFifthCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeTobTecTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeTobTecTrackCandidates.SeedProducer = cms.InputTag("iterativeTobTecSeeds","TobTecLayerPairs")
iterativeTobTecTrackCandidates.MinNumberOfCrossedLayers = 3


# track producer
#from FastSimulation.Tracking.IterativeFifthTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeTobTecTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeTobTecTracks.src = 'iterativeTobTecTrackCandidates'
iterativeTobTecTracks.TTRHBuilder = 'WithoutRefit'
iterativeTobTecTracks.Fitter = 'KFFittingSmootherFifth'
iterativeTobTecTracks.Propagator = 'PropagatorWithMaterial'
iterativeTobTecTracks.AlgorithmName = cms.string('tobTecStep')

# simtrack id producer
tobTecStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativeTobTecTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )



# track selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
tobTecStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativeTobTecTracks',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'tobTecStepLoose',
                            chi2n_par = 0.4,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 2,
                            d0_par1 = ( 2.0, 4.0 ),
                            dz_par1 = ( 1.8, 4.0 ),
                            d0_par2 = ( 2.0, 4.0 ),
                            dz_par2 = ( 1.8, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'tobTecStepTight',
                            preFilterName = 'tobTecStepLoose',
                            chi2n_par = 0.3,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 0,
                            minNumber3DLayers = 2,
                            d0_par1 = ( 1.5, 4.0 ),
                            dz_par1 = ( 1.4, 4.0 ),
                            d0_par2 = ( 1.5, 4.0 ),
                            dz_par2 = ( 1.4, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'tobTecStep',
                            preFilterName = 'tobTecStepTight',
                            chi2n_par = 0.2,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 0,
                            minNumber3DLayers = 2,
                            d0_par1 = ( 1.4, 4.0 ),
                            dz_par1 = ( 1.3, 4.0 ),
                            d0_par2 = ( 1.4, 4.0 ),
                            dz_par2 = ( 1.3, 4.0 )
                            ),
                    ) #end of vpset
            ) #end of clone

# sequence
iterativeTobTecStep = cms.Sequence(iterativeTobTecSeeds
                                      +iterativeTobTecTrackCandidates
                                      +iterativeTobTecTracks
                                      +tobTecStepSelector
                                      +tobTecStepIds)

