import FWCore.ParameterSet.Config as cms

###############################################
# Low Pt tracking using pixel-triplet seeding #
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

firstfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("preMergingFirstStepTracksWithQuality")
)

secClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("newClusters"),
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("newClusters"),
    stripClusters = cms.InputTag("newClusters"),
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
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'secClusters'
    )
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'secClusters'
    )

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
seclayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
    ComponentName = 'SecLayerTriplets'
    )
seclayertriplets.BPix.HitProducer = 'secPixelRecHits'
seclayertriplets.FPix.HitProducer = 'secPixelRecHits'


# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 17.5
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.2

# Use modified pixel-triplet code that works best for large impact parameters
#secTriplets.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
#secTriplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)

# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'secMeasurementTracker',
    pixelClusterProducer = 'secClusters',
    stripClusterProducer = 'secClusters'
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
secCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'secCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 3,
    minPt = 0.1
    )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'secCkfTrajectoryBuilder',
    MeasurementTrackerName = 'secMeasurementTracker',
    trajectoryFilterName = 'secCkfTrajectoryFilter'
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
    src = 'secTrackCandidates',
    clusterRemovalInfo = 'secClusters'
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

secStepVtxLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'secWithMaterialTracks',
    keepAllTracks = False,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 2.0,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    d0_par1 = ( 1.2, 3.0 ),
    dz_par1 = ( 1.2, 3.0 ),
    d0_par2 = ( 1.3, 3.0 ),
    dz_par2 = ( 1.3, 3.0 )
    )

secStepTrkLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'secWithMaterialTracks',
    keepAllTracks = False,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.9,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 4,
    d0_par1 = ( 1.5, 4.0 ),
    dz_par1 = ( 1.5, 4.0 ),
    d0_par2 = ( 1.5, 4.0 ),
    dz_par2 = ( 1.5, 4.0 )
    )

secStepLoose = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'secStepVtxLoose',
    TrackProducer2 = 'secStepTrkLoose'
    )


secStepVtxTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'secStepVtxLoose',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.9,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 3,
    d0_par1 = ( 0.95, 3.0 ),
    dz_par1 = ( 0.9, 3.0 ),
    d0_par2 = ( 1.0, 3.0 ),
    dz_par2 = ( 1.0, 3.0 )
    )

secStepTrkTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'secStepTrkLoose',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.7,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 5,
    minNumber3DLayers = 3,
    maxNumberLostLayers = 1,
    d0_par1 = ( 1.0, 4.0 ),
    dz_par1 = ( 1.0, 4.0 ),
    d0_par2 = ( 1.0, 4.0 ),
    dz_par2 = ( 1.0, 4.0 )
    )

secStepTight = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'secStepVtxTight',
    TrackProducer2 = 'secStepTrkTight'
    )


secStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'secStepVtxTight',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.9,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    minNumber3DLayers = 3,
    maxNumberLostLayers = 1,
    d0_par1 = ( 0.85, 3.0 ),
    dz_par1 = ( 0.8, 3.0 ),
    d0_par2 = ( 0.9, 3.0 ),
    dz_par2 = ( 0.9, 3.0 )
)

secStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'secStepTrkTight',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.5,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 5,
    minNumber3DLayers = 3,
    maxNumberLostLayers = 1,
    d0_par1 = ( 0.9, 4.0 ),
    dz_par1 = ( 0.9, 4.0 ),
    d0_par2 = ( 0.9, 4.0 ),
    dz_par2 = ( 0.9, 4.0 )
    )

secStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'secStepVtx',
    TrackProducer2 = 'secStepTrk'
    )

secondStep = cms.Sequence(firstfilter*
                          secClusters*
                          secPixelRecHits*secStripRecHits*
                          secTriplets*
                          secTrackCandidates*
                          secWithMaterialTracks*
                          secStepVtxLoose*secStepTrkLoose*secStepLoose*
                          secStepVtxTight*secStepTrkTight*secStepTight*
                          secStepVtx*secStepTrk*secStep)
