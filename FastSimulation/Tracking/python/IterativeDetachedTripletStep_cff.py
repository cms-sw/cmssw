import FWCore.ParameterSet.Config as cms

### ITERATIVE TRACKING: STEP 3 ###

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeDetachedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeDetachedTripletSeeds.simTrackSelection.skipSimTrackIdTags = [cms.InputTag("initialStepIds"), cms.InputTag("lowPtTripletStepIds"), cms.InputTag("pixelPairStepIds")]
iterativeDetachedTripletSeeds.simTrackSelection.minLayersCrossed = 3
iterativeDetachedTripletSeeds.simTrackSelection.pTMin = 0.3
iterativeDetachedTripletSeeds.simTrackSelection.maxD0 = 30. # it was 5.
iterativeDetachedTripletSeeds.simTrackSelection.maxZ0 = 50.
iterativeDetachedTripletSeeds.outputSeedCollectionName = 'DetachedPixelTriplets'
iterativeDetachedTripletSeeds.originRadius = 1.5
iterativeDetachedTripletSeeds.maxZ = 15.
iterativeDetachedTripletSeeds.originpTMin = 0.075
iterativeDetachedTripletSeeds.primaryVertex = ''

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
iterativeDetachedTripletTrackCandidates.SeedProducer = cms.InputTag("iterativeDetachedTripletSeeds",'DetachedPixelTriplets')
iterativeDetachedTripletTrackCandidates.MinNumberOfCrossedLayers = 3 

# track producer
#from FastSimulation.Tracking.IterativeSecondTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeDetachedTripletTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeDetachedTripletTracks.src = 'iterativeDetachedTripletTrackCandidates'
iterativeDetachedTripletTracks.TTRHBuilder = 'WithoutRefit'
iterativeDetachedTripletTracks.Fitter = 'KFFittingSmootherSecond'
iterativeDetachedTripletTracks.Propagator = 'PropagatorWithMaterial'
iterativeDetachedTripletTracks.AlgorithmName = cms.string('detachedTripletStep')

# simtrack id producer
detachedTripletStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativeDetachedTripletTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )


# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativeDetachedTripletTracks',
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
        TrackProducers = cms.VInputTag(cms.InputTag('iterativeDetachedTripletTracks'),
                                                                          cms.InputTag('iterativeDetachedTripletTracks')),
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
                                            detachedTripletStepIds+
                                            detachedTripletStepSelector+
                                            detachedTripletStep)
