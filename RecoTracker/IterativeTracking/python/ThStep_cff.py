import FWCore.ParameterSet.Config as cms

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#TRACKER HITS
thPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
thStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
thPixelRecHits.src = 'thClusters'
thStripRecHits.ClusterProducer = 'thClusters'

# Propagator taking into account momentum uncertainty in multiple
# scattering calculation.

import TrackingTools.MaterialEffects.MaterialPropagator_cfi
MaterialPropagatorPtMin03 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone()
MaterialPropagatorPtMin03.ComponentName = 'PropagatorWithMaterialPtMin03'
MaterialPropagatorPtMin03.ptMin = 0.3

import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin03 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone()
OppositeMaterialPropagatorPtMin03.ComponentName = 'PropagatorWithMaterialOppositePtMin03'
OppositeMaterialPropagatorPtMin03.ptMin = 0.3

import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
#SEEDS
thPLSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
thPLSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'ThLayerPairs'
thPLSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
thPLSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 7.0
thPLSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.2
import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi
thPLSeeds.SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi.SeedFromConsecutiveHitsStraightLineCreator.clone(
    propagator = cms.string('PropagatorWithMaterialPtMin03')
    )


#TRAJECTORY MEASUREMENT
thMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
thMeasurementTracker.ComponentName = 'thMeasurementTracker'
thMeasurementTracker.pixelClusterProducer = 'thClusters'
thMeasurementTracker.stripClusterProducer = 'thClusters'

#TRAJECTORY FILTER
thCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
thCkfTrajectoryFilter.ComponentName = 'thCkfTrajectoryFilter'
thCkfTrajectoryFilter.filterPset.maxLostHits = 0
thCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
thCkfTrajectoryFilter.filterPset.minPt = 0.3

#TRAJECTORY BUILDER
thCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
thCkfTrajectoryBuilder.ComponentName = 'thCkfTrajectoryBuilder'
thCkfTrajectoryBuilder.MeasurementTrackerName = 'thMeasurementTracker'
thCkfTrajectoryBuilder.trajectoryFilterName = 'thCkfTrajectoryFilter'
thCkfTrajectoryBuilder.propagatorAlong = cms.string('PropagatorWithMaterialPtMin03')
thCkfTrajectoryBuilder.propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin03')


#TRACK CANDIDATES
thTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
thTrackCandidates.src = cms.InputTag('thPLSeeds')
thTrackCandidates.TrajectoryBuilder = 'thCkfTrajectoryBuilder'
thTrackCandidates.doSeedingRegionRebuilding = True
thTrackCandidates.useHitsSplitting = True


#TRACKS
thWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
thWithMaterialTracks.AlgorithmName = cms.string('iter3')
thWithMaterialTracks.src = 'thTrackCandidates'
thWithMaterialTracks.clusterRemovalInfo = 'thClusters'


secfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("secStep")
)

#HIT REMOVAL
thClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secClusters"),
    trajectories = cms.InputTag("secfilter"),
    pixelClusters = cms.InputTag("secClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("secClusters")
)

#SEEDING LAYERS
thlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    ComponentName = cms.string('ThLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix2+BPix3',
        'BPix1+FPix1_pos',
        'BPix1+FPix1_neg',
        'FPix1_pos+FPix2_pos',
        'FPix1_neg+FPix2_neg',
        'FPix2_pos+TEC2_pos',
        'FPix2_neg+TEC2_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

# track selection
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

thStepVtxLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
thStepVtxLoose.src = 'thWithMaterialTracks'
thStepVtxLoose.keepAllTracks = False
thStepVtxLoose.copyExtras = True
thStepVtxLoose.copyTrajectories = True
thStepVtxLoose.chi2n_par = 2.0
thStepVtxLoose.res_par = ( 0.003, 0.001 )
thStepVtxLoose.d0_par1 = ( 1.2, 3.0 )
thStepVtxLoose.dz_par1 = ( 1.2, 3.0 )
thStepVtxLoose.d0_par2 = ( 1.3, 3.0 )
thStepVtxLoose.dz_par2 = ( 1.3, 3.0 )

thStepTrkLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
thStepTrkLoose.src = 'thWithMaterialTracks'
thStepTrkLoose.keepAllTracks = False
thStepTrkLoose.copyExtras = True
thStepTrkLoose.copyTrajectories = True
thStepTrkLoose.chi2n_par = 0.9
thStepTrkLoose.res_par = ( 0.003, 0.001 )
thStepTrkLoose.minNumberLayers = 4
thStepTrkLoose.d0_par1 = ( 1.8, 4.0 )
thStepTrkLoose.dz_par1 = ( 1.8, 4.0 )
thStepTrkLoose.d0_par2 = ( 1.8, 4.0 )
thStepTrkLoose.dz_par2 = ( 1.8, 4.0 )

thStepLoose = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
thStepLoose.TrackProducer1 = 'thStepVtxLoose'
thStepLoose.TrackProducer2 = 'thStepTrkLoose'


thStepVtxTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
thStepVtxTight.src = 'thStepVtxLoose'
thStepVtxTight.keepAllTracks = True
thStepVtxTight.copyExtras = True
thStepVtxTight.copyTrajectories = True
thStepVtxTight.chi2n_par = 0.9
thStepVtxTight.res_par = ( 0.003, 0.001 )
thStepVtxTight.d0_par1 = ( 1.0, 3.0 )
thStepVtxTight.dz_par1 = ( 1.0, 3.0 )
thStepVtxTight.d0_par2 = ( 1.1, 3.0 )
thStepVtxTight.dz_par2 = ( 1.1, 3.0 )

thStepTrkTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
thStepTrkTight.src = 'thStepTrkLoose'
thStepTrkTight.keepAllTracks = True
thStepTrkTight.copyExtras = True
thStepTrkTight.copyTrajectories = True
thStepTrkTight.chi2n_par = 0.7
thStepTrkTight.res_par = ( 0.003, 0.001 )
thStepTrkTight.minNumberLayers = 5
thStepTrkTight.d0_par1 = ( 1.1, 4.0 )
thStepTrkTight.dz_par1 = ( 1.1, 4.0 )
thStepTrkTight.d0_par2 = ( 1.1, 4.0 )
thStepTrkTight.dz_par2 = ( 1.1, 4.0 )

thStepTight = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
thStepTight.TrackProducer1 = 'thStepVtxTight'
thStepTight.TrackProducer2 = 'thStepTrkTight'


thStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
thStepVtx.src = 'thStepVtxTight'
thStepVtx.keepAllTracks = True
thStepVtx.copyExtras = True
thStepVtx.copyTrajectories = True
thStepVtx.chi2n_par = 0.9
thStepVtx.res_par = ( 0.003, 0.001 )
thStepVtx.d0_par1 = ( 0.9, 3.0 )
thStepVtx.dz_par1 = ( 0.9, 3.0 )
thStepVtx.d0_par2 = ( 1.0, 3.0 )
thStepVtx.dz_par2 = ( 1.0, 3.0 )

thStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
thStepTrk.src = 'thStepTrkTight'
thStepTrk.keepAllTracks = True
thStepTrk.copyExtras = True
thStepTrk.copyTrajectories = True
thStepTrk.chi2n_par = 0.5
thStepTrk.res_par = ( 0.003, 0.001 )
thStepTrk.minNumberLayers = 5
thStepTrk.d0_par1 = ( 1.0, 4.0 )
thStepTrk.dz_par1 = ( 1.0, 4.0 )
thStepTrk.d0_par2 = ( 1.0, 4.0 )
thStepTrk.dz_par2 = ( 1.0, 4.0 )

thStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
thStep.TrackProducer1 = 'thStepVtx'
thStep.TrackProducer2 = 'thStepTrk'

thirdStep = cms.Sequence(secfilter*
                         thClusters*
                         thPixelRecHits*thStripRecHits*
                         thPLSeeds*
                         thTrackCandidates*
                         thWithMaterialTracks*
                         thStepVtxLoose*thStepTrkLoose*thStepLoose*
                         thStepVtxTight*thStepTrkTight*thStepTight*
                         thStepVtx*thStepTrk*thStep)
