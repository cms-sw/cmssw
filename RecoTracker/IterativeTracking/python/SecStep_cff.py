import FWCore.ParameterSet.Config as cms

###############################################
# Low Pt tracking using pixel-triplet seeding #
###############################################

secClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution = cms.bool(True),
    oldClusterRemovalInfo = cms.InputTag("newClusters"),
    trajectories = cms.InputTag("preFilterStepOneTracks"),
    overrideTrkQuals = cms.InputTag('firstSelector','preMergingFirstStepTracksWithQuality'),                         
    TrackQuality = cms.string('highPurity'),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
                           
# For debug purposes, you can run this iteration not eliminating any hits from previous ones by
# instead using
#    trajectories = cms.InputTag("zeroStepFilter"),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),
#     Common = cms.PSet(
#       maxChi2 = cms.double(0.0)
#    )
                           
)

# TRACKER HITS
#import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#secPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
#    src = 'secClusters'
#    )
#import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
#secStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
#    ClusterProducer = 'secClusters'
#    )

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
seclayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
    ComponentName = 'SecLayerTriplets'
    )
seclayertriplets.BPix.HitProducer = 'siPixelRecHits'
seclayertriplets.BPix.skipClusters = cms.InputTag('secClusters')
seclayertriplets.FPix.HitProducer = 'siPixelRecHits'
seclayertriplets.FPix.skipClusters = cms.InputTag('secClusters')


# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.075,
    nSigmaZ = 3.3
    )
    )
    )
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.ClusterCheckPSet.PixelClusterCollectionLabel = 'siPixelClusters'
secTriplets.ClusterCheckPSet.ClusterCollectionLabel = 'siStripClusters'
      

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
secTriplets.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# Use modified pixel-triplet code that works best for large impact parameters
#secTriplets.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
#secTriplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)

# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'secMeasurementTracker',
    skipClusters = cms.InputTag('secClusters'),
    pixelClusterProducer = 'siPixelClusters',
    stripClusterProducer = 'siStripClusters'
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
secCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'secCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 3,
    minPt = 0.075
    )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'secCkfTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'secCkfTrajectoryFilter',
    clustersToSkip = cms.InputTag('secClusters')
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('secTriplets'),
    TrajectoryBuilder = 'secCkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
secWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter2'),
    src = 'secTrackCandidates'
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
secSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='secWithMaterialTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'secStepVtxLoose',
            chi2n_par = 1.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
            ), #end of pset for thStepVtxLoose
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'secStepTrkLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.6, 4.0 ),
            dz_par1 = ( 1.6, 4.0 ),
            d0_par2 = ( 1.6, 4.0 ),
            dz_par2 = ( 1.6, 4.0 )
            ), #end of pset for thStepTrkLoose
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'secStepVtxTight',
            preFilterName = 'secStepVtxLoose',
            chi2n_par = 0.7,
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
            name = 'secStepTrkTight',
            preFilterName = 'secStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'secStepVtx',
            preFilterName = 'secStepVtxTight',
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
            name = 'secStepTrk',
            preFilterName = 'secStepTrkTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            )
        ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
secStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('secWithMaterialTracks'),cms.InputTag('secWithMaterialTracks')),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("secSelector","secStepVtx"),cms.InputTag("secSelector","secStepTrk")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
)                        


#import RecoTracker.FinalTrackSelectors.trackQualMerger_cfi
#secQualMerger = RecoTracker.FinalTrackSelectors.trackQualMerger_cfi.trackQualMerger.clone()

#secQualMerger = cms.EDProducer("TrackQualMerger",
#                               src=cms.InputTag('secWithMaterialTracks'),
#                               trackSelectors=cms.VInputTag(cms.InputTag("secSelector","secStepVtx"),cms.InputTag("secSelector","secStepTrk"))
#                               )



secondStep = cms.Sequence(secClusters*
                          secTriplets*
                          secTrackCandidates*
                          secWithMaterialTracks*
                          secSelector*secStep)
