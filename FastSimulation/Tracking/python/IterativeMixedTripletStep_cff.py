import FWCore.ParameterSet.Config as cms

# step 3

# seeding
#from FastSimulation.Tracking.IterativeMixedTripletStepSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeMixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeMixedTripletStepSeeds.firstHitSubDetectorNumber = [2]
##iterativeMixedTripletStepSeeds.firstHitSubDetectors = [1, 2, 6]
iterativeMixedTripletStepSeeds.firstHitSubDetectors = [1, 2]
iterativeMixedTripletStepSeeds.secondHitSubDetectorNumber = [3]
iterativeMixedTripletStepSeeds.secondHitSubDetectors = [1, 2, 6]
iterativeMixedTripletStepSeeds.thirdHitSubDetectorNumber = [0]
iterativeMixedTripletStepSeeds.thirdHitSubDetectors = []
iterativeMixedTripletStepSeeds.seedingAlgo = ['MixedTriplets']
iterativeMixedTripletStepSeeds.minRecHits = [3]
iterativeMixedTripletStepSeeds.pTMin = [0.15]
iterativeMixedTripletStepSeeds.maxD0 = [10.]
iterativeMixedTripletStepSeeds.maxZ0 = [30.]
iterativeMixedTripletStepSeeds.numberOfHits = [2]
iterativeMixedTripletStepSeeds.originRadius = [2.0] # was 1.2
iterativeMixedTripletStepSeeds.originHalfLength = [10.0] # was 7.0
iterativeMixedTripletStepSeeds.originpTMin = [0.35] # we need to add another seed for endcaps only, with 0.5
iterativeMixedTripletStepSeeds.zVertexConstraint = [-1.0]
iterativeMixedTripletStepSeeds.primaryVertices = ['none']


# candidate producer
#from FastSimulation.Tracking.IterativeThirdCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeMixedTripletStepCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeMixedTripletStepCandidates.SeedProducer = cms.InputTag("iterativeMixedTripletStepSeeds","MixedTriplets")
iterativeMixedTripletStepCandidates.TrackProducers = ['pixelPairStepTracks', 'detachedTripletStepTracks']
iterativeMixedTripletStepCandidates.KeepFittedTracks = False
iterativeMixedTripletStepCandidates.MinNumberOfCrossedLayers = 3


# track producer
#from FastSimulation.Tracking.IterativeThirdTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeMixedTripletStepTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeMixedTripletStepTracks.src = 'iterativeMixedTripletStepCandidates'
iterativeMixedTripletStepTracks.TTRHBuilder = 'WithoutRefit'
##iterativeMixedTripletStepTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeMixedTripletStepTracks.Fitter = 'KFFittingSmootherThird'
iterativeMixedTripletStepTracks.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeMixedTripletStepMerger_cfi import *
mixedTripletStepTracks = cms.EDProducer("FastTrackMerger",
                                        TrackProducers = cms.VInputTag(cms.InputTag("iterativeMixedTripletStepCandidates"),
                                                                       cms.InputTag("iterativeMixedTripletStepTracks")),
                                        RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStepTracks"),
                                                                                        cms.InputTag("lowPtTripletStepTracks"),   
                                                                                        cms.InputTag("pixelPairStepTracks"),   
                                                                                        cms.InputTag("detachedTripletStepTracks")),    
                                        trackAlgo = cms.untracked.uint32(8),
                                        MinNumberOfTrajHits = cms.untracked.uint32(4), # ?
                                        MaxLostTrajHits = cms.untracked.uint32(0)                                          
                                        )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
mixedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='mixedTripletStepTracks',
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

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
mixedTripletStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('mixedTripletStepTracks'),
                                   cms.InputTag('mixedTripletStepTracks')),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("mixedTripletStepSelector","mixedTripletStepVtx"),
                                       cms.InputTag("mixedTripletStepSelector","mixedTripletStepTrk")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
    )

# sequence
iterativeMixedTripletStep = cms.Sequence(iterativeMixedTripletStepSeeds+
                                         iterativeMixedTripletStepCandidates+
                                         iterativeMixedTripletStepTracks+
                                         mixedTripletStepTracks+
                                         mixedTripletStepSelector+
                                         mixedTripletStep)

