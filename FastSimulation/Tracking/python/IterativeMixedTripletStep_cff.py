import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeMixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeMixedTripletStepSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepIds"), 
    cms.InputTag("detachedTripletStepIds"), 
    cms.InputTag("lowPtTripletStepIds"), 
    cms.InputTag("pixelPairStepIds")]
]
iterativeMixedTripletStepSeeds.simTrackSelection.pTMin = 0.15
iterativeMixedTripletStepSeeds.simTrackSelection.maxD0 = 10.
iterativeMixedTripletStepSeeds.simTrackSelection.maxZ0 = 30.
iterativeMixedTripletStepSeeds.minLayersCrossed = 3
iterativeMixedTripletStepSeeds.originRadius = 2.0
iterativeMixedTripletStepSeeds.originHalfLength = 10.0
iterativeMixedTripletStepSeeds.originpTMin = 0.35
iterativeMixedTripletStepSeeds.primaryVertex = ''
# combine both (A&B); Note: in FullSim, different cuts are applied for A & B seeds; 
# in FastSim there is only one cut set, which is tuned
# probably better to change this
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSeedLayersA,mixedTripletStepSeedLayersB
iterativeMixedTripletStepSeeds.layerList = mixedTripletStepSeedLayersA.layerList+mixedTripletStepSeedLayersB.layerList

# track candidates

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeMixedTripletStepCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeMixedTripletStepCandidates.SeedProducer = cms.InputTag("iterativeMixedTripletStepSeeds")
iterativeMixedTripletStepCandidates.MinNumberOfCrossedLayers = 3


# tracks

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeMixedTripletStepTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeMixedTripletStepTracks.src = 'iterativeMixedTripletStepCandidates'
iterativeMixedTripletStepTracks.TTRHBuilder = 'WithoutRefit'
iterativeMixedTripletStepTracks.Fitter = 'KFFittingSmootherThird'
iterativeMixedTripletStepTracks.Propagator = 'PropagatorWithMaterial'
iterativeMixedTripletStepTracks.AlgorithmName = cms.string('mixedTripletStep')

# track identification

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
mixedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativeMixedTripletStepTracks',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'mixedTripletStepVtxLoose',
                            chi2n_par = 1.2,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 3,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 2,
                            d0_par1 = ( 1.2, 3.0 ),
                            dz_par1 = ( 1.2, 3.0 ),
                            d0_par2 = ( 1.3, 3.0 ),
                            dz_par2 = ( 1.3, 3.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'mixedTripletStepTrkLoose',
                            chi2n_par = 0.6,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 4,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1.2, 4.0 ),
                            dz_par1 = ( 1.2, 4.0 ),
                            d0_par2 = ( 1.2, 4.0 ),
                            dz_par2 = ( 1.2, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'mixedTripletStepVtxTight',
                            preFilterName = 'mixedTripletStepVtxLoose',
                            chi2n_par = 0.6,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 3,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1.1, 3.0 ),
                            dz_par1 = ( 1.1, 3.0 ),
                            d0_par2 = ( 1.2, 3.0 ),
                            dz_par2 = ( 1.2, 3.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'mixedTripletStepTrkTight',
                            preFilterName = 'mixedTripletStepTrkLoose',
                            chi2n_par = 0.4,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 4,
                            d0_par1 = ( 1.1, 4.0 ),
                            dz_par1 = ( 1.1, 4.0 ),
                            d0_par2 = ( 1.1, 4.0 ),
                            dz_par2 = ( 1.1, 4.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'mixedTripletStepVtx',
                            preFilterName = 'mixedTripletStepVtxTight',
                            chi2n_par = 0.4,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 3,
                            maxNumberLostLayers = 1,
                            minNumber3DLayers = 3,
                            d0_par1 = ( 1.1, 3.0 ),
                            dz_par1 = ( 1.1, 3.0 ),
                            d0_par2 = ( 1.2, 3.0 ),
                            dz_par2 = ( 1.2, 3.0 )
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'mixedTripletStepTrk',
                            preFilterName = 'mixedTripletStepTrkTight',
                            chi2n_par = 0.3,
                            res_par = ( 0.003, 0.001 ),
                            minNumberLayers = 5,
                            maxNumberLostLayers = 0,
                            minNumber3DLayers = 4,
                            d0_par1 = ( 0.9, 4.0 ),
                            dz_par1 = ( 0.9, 4.0 ),
                            d0_par2 = ( 0.9, 4.0 ),
                            dz_par2 = ( 0.9, 4.0 )
                            )
                    ) #end of vpset
            ) #end of clone

# unused collection?

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
mixedTripletStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('iterativeMixedTripletStepTracks'),
                                   cms.InputTag('iterativeMixedTripletStepTracks')),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("mixedTripletStepSelector","mixedTripletStepVtx"),
                                       cms.InputTag("mixedTripletStepSelector","mixedTripletStepTrk")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
    )

# simtrack id producer

mixedTripletStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativeMixedTripletStepTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# sequence
iterativeMixedTripletStep = cms.Sequence(iterativeMixedTripletStepSeeds+
                                         iterativeMixedTripletStepCandidates+
                                         iterativeMixedTripletStepTracks+
                                         mixedTripletStepIds+
                                         mixedTripletStepSelector+
                                         mixedTripletStep)

