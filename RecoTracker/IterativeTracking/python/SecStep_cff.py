import FWCore.ParameterSet.Config as cms

firstfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("preMergingFirstStepTracksWithQuality")
)


# new hit collection
secClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("newClusters"),
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("newClusters"),
    stripClusters = cms.InputTag("newClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()

secPixelRecHits.src = cms.InputTag("secClusters")
secStripRecHits.ClusterProducer = 'secClusters'


# seeding

import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
seclayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone()
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff
secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff.globalSeedsFromTripletsWithVertices.clone()

seclayertriplets.ComponentName = 'SecLayerTriplets'
seclayertriplets.BPix.HitProducer = 'secPixelRecHits'
seclayertriplets.FPix.HitProducer = 'secPixelRecHits'
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 17.5
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3


# building 
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
secCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()

secMeasurementTracker.ComponentName = 'secMeasurementTracker'
secMeasurementTracker.pixelClusterProducer = 'secClusters'
secMeasurementTracker.stripClusterProducer = 'secClusters'

secCkfTrajectoryFilter.ComponentName = 'secCkfTrajectoryFilter'
secCkfTrajectoryFilter.filterPset.maxLostHits = 1
secCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
secCkfTrajectoryFilter.filterPset.minPt = 0.3
secCkfTrajectoryBuilder.ComponentName = 'secCkfTrajectoryBuilder'
secCkfTrajectoryBuilder.MeasurementTrackerName = 'secMeasurementTracker'
secCkfTrajectoryBuilder.trajectoryFilterName = 'secCkfTrajectoryFilter'

secTrackCandidates.src = cms.InputTag('secTriplets')
secTrackCandidates.TrajectoryBuilder = 'secCkfTrajectoryBuilder'
secTrackCandidates.doSeedingRegionRebuilding = True
secTrackCandidates.useHitsSplitting = True


# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
secWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
secWithMaterialTracks.AlgorithmName = cms.string('iter2')
secWithMaterialTracks.src = 'secTrackCandidates'
secWithMaterialTracks.clusterRemovalInfo = 'secClusters'


# track selection
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

secStepVtxLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
secStepVtxLoose.src = 'secWithMaterialTracks'
secStepVtxLoose.keepAllTracks = False
secStepVtxLoose.copyExtras = True
secStepVtxLoose.copyTrajectories = True
secStepVtxLoose.chi2n_par = 2.0
secStepVtxLoose.res_par = ( 0.003, 0.001 )
secStepVtxLoose.d0_par1 = ( 1.2, 3.0 )
secStepVtxLoose.dz_par1 = ( 1.2, 3.0 )
secStepVtxLoose.d0_par2 = ( 1.3, 3.0 )
secStepVtxLoose.dz_par2 = ( 1.3, 3.0 )

secStepTrkLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
secStepTrkLoose.src = 'secWithMaterialTracks'
secStepTrkLoose.keepAllTracks = False
secStepTrkLoose.copyExtras = True
secStepTrkLoose.copyTrajectories = True
secStepTrkLoose.chi2n_par = 0.9
secStepTrkLoose.res_par = ( 0.003, 0.001 )
secStepTrkLoose.minNumberLayers = 4
secStepTrkLoose.d0_par1 = ( 1.5, 4.0 )
secStepTrkLoose.dz_par1 = ( 1.5, 4.0 )
secStepTrkLoose.d0_par2 = ( 1.5, 4.0 )
secStepTrkLoose.dz_par2 = ( 1.5, 4.0 )

secStepLoose = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
secStepLoose.TrackProducer1 = 'secStepVtxLoose'
secStepLoose.TrackProducer2 = 'secStepTrkLoose'


secStepVtxTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
secStepVtxTight.src = 'secStepVtxLoose'
secStepVtxTight.keepAllTracks = True
secStepVtxTight.copyExtras = True
secStepVtxTight.copyTrajectories = True
secStepVtxTight.chi2n_par = 0.9
secStepVtxTight.res_par = ( 0.003, 0.001 )
secStepVtxTight.d0_par1 = ( 0.95, 3.0 )
secStepVtxTight.dz_par1 = ( 0.9, 3.0 )
secStepVtxTight.d0_par2 = ( 1.0, 3.0 )
secStepVtxTight.dz_par2 = ( 1.0, 3.0 )

secStepTrkTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
secStepTrkTight.src = 'secStepTrkLoose'
secStepTrkTight.keepAllTracks = True
secStepTrkTight.copyExtras = True
secStepTrkTight.copyTrajectories = True
secStepTrkTight.chi2n_par = 0.7
secStepTrkTight.res_par = ( 0.003, 0.001 )
secStepTrkTight.minNumberLayers = 5
secStepTrkTight.d0_par1 = ( 1.0, 4.0 )
secStepTrkTight.dz_par1 = ( 1.0, 4.0 )
secStepTrkTight.d0_par2 = ( 1.0, 4.0 )
secStepTrkTight.dz_par2 = ( 1.0, 4.0 )

secStepTight = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
secStepTight.TrackProducer1 = 'secStepVtxTight'
secStepTight.TrackProducer2 = 'secStepTrkTight'


secStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
secStepVtx.src = 'secStepVtxTight'
secStepVtx.keepAllTracks = True
secStepVtx.copyExtras = True
secStepVtx.copyTrajectories = True
secStepVtx.chi2n_par = 0.9
secStepVtx.res_par = ( 0.003, 0.001 )
secStepVtx.d0_par1 = ( 0.85, 3.0 )
secStepVtx.dz_par1 = ( 0.8, 3.0 )
secStepVtx.d0_par2 = ( 0.9, 3.0 )
secStepVtx.dz_par2 = ( 0.9, 3.0 )

secStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
secStepTrk.src = 'secStepTrkTight'
secStepTrk.keepAllTracks = True
secStepTrk.copyExtras = True
secStepTrk.copyTrajectories = True
secStepTrk.chi2n_par = 0.5
secStepTrk.res_par = ( 0.003, 0.001 )
secStepTrk.minNumberLayers = 5
secStepTrk.d0_par1 = ( 0.9, 4.0 )
secStepTrk.dz_par1 = ( 0.9, 4.0 )
secStepTrk.d0_par2 = ( 0.9, 4.0 )
secStepTrk.dz_par2 = ( 0.9, 4.0 )

secStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
secStep.TrackProducer1 = 'secStepVtx'
secStep.TrackProducer2 = 'secStepTrk'

secondStep = cms.Sequence(firstfilter*
                          secClusters*
                          secPixelRecHits*secStripRecHits*
                          secTriplets*
                          secTrackCandidates*
                          secWithMaterialTracks*
                          secStepVtxLoose*secStepTrkLoose*secStepLoose*
                          secStepVtxTight*secStepTrkTight*secStepTight*
                          secStepVtx*secStepTrk*secStep)
