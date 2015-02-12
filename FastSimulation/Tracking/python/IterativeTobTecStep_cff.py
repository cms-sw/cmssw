import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeTobTecSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeTobTecSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepIds"), 
    cms.InputTag("detachedTripletStepIds"), 
    cms.InputTag("lowPtTripletStepIds"), 
    cms.InputTag("pixelPairStepIds"), 
    cms.InputTag("mixedTripletStepIds"), 
    cms.InputTag("pixelLessStepIds")]
iterativeTobTecSeeds.simTrackSelection.pTMin = 0.3
iterativeTobTecSeeds.simTrackSelection.maxD0 = 99.
iterativeTobTecSeeds.simTrackSelection.maxZ0 = 99.
iterativeTobTecSeeds.minLayersCrossed = 4
iterativeTobTecSeeds.originRadius = 6.0
iterativeTobTecSeeds.originHalfLength = 30.0
iterativeTobTecSeeds.originpTMin = 0.6

iterativeTobTecSeeds.primaryVertex = ''

from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair
iterativeTobTecSeeds.layerList = ['TOB1+TOB2']
iterativeTobTecSeeds.layerList.extend(tobTecStepSeedLayersPair.layerList)

# track candidates

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeTobTecTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeTobTecTrackCandidates.SeedProducer = cms.InputTag("iterativeTobTecSeeds")
iterativeTobTecTrackCandidates.MinNumberOfCrossedLayers = 3


# tracks

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeTobTecTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeTobTecTracks.src = 'iterativeTobTecTrackCandidates'
iterativeTobTecTracks.TTRHBuilder = 'WithoutRefit'
iterativeTobTecTracks.Fitter = 'KFFittingSmootherFifth'
iterativeTobTecTracks.Propagator = 'PropagatorWithMaterial'
iterativeTobTecTracks.AlgorithmName = cms.string('tobTecStep')

# track identification

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

# simtrack id producer

tobTecStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativeTobTecTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# final sequence
iterativeTobTecStep = cms.Sequence(iterativeTobTecSeeds
                                      +iterativeTobTecTrackCandidates
                                      +iterativeTobTecTracks
                                      +tobTecStepSelector
                                      +tobTecStepIds)

