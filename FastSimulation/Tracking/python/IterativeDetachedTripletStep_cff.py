import FWCore.ParameterSet.Config as cms

# step 2

# seeding
#from FastSimulation.Tracking.IterativeSecondSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeDetachedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeDetachedTripletSeeds.firstHitSubDetectorNumber = [1]
iterativeDetachedTripletSeeds.firstHitSubDetectors = [1]
iterativeDetachedTripletSeeds.secondHitSubDetectorNumber = [2]
iterativeDetachedTripletSeeds.secondHitSubDetectors = [1, 2]
iterativeDetachedTripletSeeds.thirdHitSubDetectorNumber = [2]
iterativeDetachedTripletSeeds.thirdHitSubDetectors = [1, 2]
iterativeDetachedTripletSeeds.seedingAlgo = ['DetachedPixelTriplets']
iterativeDetachedTripletSeeds.minRecHits = [3]
iterativeDetachedTripletSeeds.pTMin = [0.3]
iterativeDetachedTripletSeeds.maxD0 = [30.] # it was 5.
iterativeDetachedTripletSeeds.maxZ0 = [50.]
iterativeDetachedTripletSeeds.numberOfHits = [3]
iterativeDetachedTripletSeeds.originRadius = [1.5] 
iterativeDetachedTripletSeeds.originHalfLength = [15.] 
iterativeDetachedTripletSeeds.originpTMin = [0.075] 
iterativeDetachedTripletSeeds.zVertexConstraint = [-1.0]
iterativeDetachedTripletSeeds.primaryVertices = ['none']

iterativeDetachedTripletSeeds.newSyntax = True
#iterativeDetachedTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeDetachedTripletSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer
#from FastSimulation.Tracking.IterativeSecondCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeDetachedTripletTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeDetachedTripletTrackCandidates.SeedProducer = cms.InputTag("iterativeDetachedTripletSeeds","DetachedPixelTriplets")
iterativeDetachedTripletTrackCandidates.TrackProducers = ['pixelPairStepTracks']
iterativeDetachedTripletTrackCandidates.KeepFittedTracks = False
iterativeDetachedTripletTrackCandidates.MinNumberOfCrossedLayers = 3 

# track producer
#from FastSimulation.Tracking.IterativeSecondTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeDetachedTripletTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeDetachedTripletTracks.src = 'iterativeDetachedTripletTrackCandidates'
iterativeDetachedTripletTracks.TTRHBuilder = 'WithoutRefit'
iterativeDetachedTripletTracks.Fitter = 'KFFittingSmootherSecond'
iterativeDetachedTripletTracks.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeSecondTrackMerger_cfi import *
detachedTripletStepTracks = cms.EDProducer("FastTrackMerger",
                                           TrackProducers = cms.VInputTag(cms.InputTag("iterativeDetachedTripletTrackCandidates"),
                                                                          cms.InputTag("iterativeDetachedTripletTracks")),
                                           RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStepTracks"), 
                                                                                           cms.InputTag("lowPtTripletStepTracks"),
                                                                                           cms.InputTag("pixelPairStepTracks")),
                                           trackAlgo = cms.untracked.uint32(7), # iter3 
                                           MinNumberOfTrajHits = cms.untracked.uint32(3),
                                           MaxLostTrajHits = cms.untracked.uint32(1)
                                           )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='detachedTripletStepTracks',
        trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepVtxLoose',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.1, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.2, 3.0 ),
            dz_par2 = ( 1.2, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.4, 4.0 ),
            dz_par1 = ( 1.4, 4.0 ),
            d0_par2 = ( 1.4, 4.0 ),
            dz_par2 = ( 1.4, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepVtxTight',
            preFilterName = 'detachedTripletStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.95, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepTrkTight',
            preFilterName = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.5,
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
            name = 'detachedTripletStepVtx',
            preFilterName = 'detachedTripletStepVtxTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.85, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepTrk',
            preFilterName = 'detachedTripletStepTrkTight',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            )
        ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
detachedTripletStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
        TrackProducers = cms.VInputTag(cms.InputTag('detachedTripletStepTracks'),
                                                                          cms.InputTag('detachedTripletStepTracks')),
            hasSelector=cms.vint32(1,1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedTripletStepSelector","detachedTripletStepVtx"),
                                                                                      cms.InputTag("detachedTripletStepSelector","detachedTripletStepTrk")),
            setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
            writeOnlyTrkQuals=cms.bool(True)
        )



# sequence
iterativeDetachedTripletStep = cms.Sequence(iterativeDetachedTripletSeeds+
                                            iterativeDetachedTripletTrackCandidates+
                                            iterativeDetachedTripletTracks+
                                            detachedTripletStepTracks+
                                            detachedTripletStepSelector+
                                            detachedTripletStep)
