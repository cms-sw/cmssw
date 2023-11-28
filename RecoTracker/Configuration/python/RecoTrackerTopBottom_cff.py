import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SubCollectionProducers.ClusterSelectorTopBottom_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import combinatorialcosmicseedfinderP5, combinatorialcosmicseedingtripletsP5, combinatorialcosmicseedingpairsTOBP5, combinatorialcosmicseedingpairsTECposP5, combinatorialcosmicseedingpairsTECnegP5
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cff import simpleCosmicBONSeeds, simpleCosmicBONSeedingLayers
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import globalCombinedSeeds
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import MeasurementTracker
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import GroupedCkfTrajectoryBuilderP5
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import ckfTrackCandidatesP5
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import ctfWithMaterialTracksCosmics
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import cosmicseedfinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmicCandidateFinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmictrackfinderCosmics


siPixelRecHitsTop = siPixelRecHits.clone(src = 'siPixelClustersTop')
siPixelRecHitsBottom = siPixelRecHits.clone(src = 'siPixelClustersBottom')
siStripMatchedRecHitsTop = siStripMatchedRecHits.clone(ClusterProducer = 'siStripClustersTop')
siStripMatchedRecHitsBottom = siStripMatchedRecHits.clone(ClusterProducer = 'siStripClustersBottom')

from RecoLocalTracker.SubCollectionProducers.TopBottomClusterInfoProducer_cfi import topBottomClusterInfoProducer
topBottomClusterInfoProducerTop = topBottomClusterInfoProducer.clone()
topBottomClusterInfoProducerBottom = topBottomClusterInfoProducer.clone(
    stripClustersNew   = 'siStripClustersBottom',
    pixelClustersNew   = 'siPixelClustersBottom',
    stripMonoHitsNew   = 'siStripMatchedRecHitsBottom:rphiRecHit',
    stripStereoHitsNew = 'siStripMatchedRecHitsBottom:stereoRecHit',
    pixelHitsNew       = 'siPixelRecHitsBottom'
)
###LOCAL RECO TASK and SEQUENCE
trackerlocalrecoTopTask = cms.Task(siPixelClustersTop,
                                          siPixelRecHitsTop,
                                          siStripClustersTop,
                                          siStripMatchedRecHitsTop,
                                          topBottomClusterInfoProducerTop)
trackerlocalrecoTop = cms.Sequence(trackerlocalrecoTopTask)
trackerlocalrecoBottomTask = cms.Task(siPixelClustersBottom,
                                          siPixelRecHitsBottom,
                                          siStripClustersBottom,
                                          siStripMatchedRecHitsBottom,
                                          topBottomClusterInfoProducerBottom)
trackerlocalrecoBottom = cms.Sequence(trackerlocalrecoBottomTask)

###CKF TOP
combinatorialcosmicseedingtripletsP5Top = combinatorialcosmicseedingtripletsP5.clone(
    TIB  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTIB = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TOB  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTOB = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TEC  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
                rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit')
)

combinatorialcosmicseedingpairsTOBP5Top = combinatorialcosmicseedingpairsTOBP5.clone(
    TIB  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTIB = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TOB  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTOB = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TEC  = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
                rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit')
)

combinatorialcosmicseedingpairsTECposP5Top = combinatorialcosmicseedingpairsTECposP5.clone(
    TEC = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
               rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit')
)

combinatorialcosmicseedingpairsTECnegP5Top = combinatorialcosmicseedingpairsTECnegP5.clone(
    TEC = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
               rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit')
)

combinatorialcosmicseedfinderP5Top = combinatorialcosmicseedfinderP5.clone(
    SeedsFromPositiveY      = True,
    SeedsFromNegativeY      = False,
    ClusterCollectionLabel  = 'siStripClustersTop',
    MaxNumberOfStripClusters = 150,

    OrderedHitsFactoryPSets = {0: dict(LayerSrc = "combinatorialcosmicseedingtripletsP5Top"),
                               1: dict(LayerSrc = "combinatorialcosmicseedingpairsTOBP5Top"),
                               2: dict(LayerSrc = "combinatorialcosmicseedingpairsTECposP5Top"),
                               3: dict(LayerSrc = "combinatorialcosmicseedingpairsTECposP5Top"),
                               4: dict(LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Top"),
                               5: dict(LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Top")
    }

)

simpleCosmicBONSeedingLayersTop = simpleCosmicBONSeedingLayers.clone(
    TIB   = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTIB  = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TOB   = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit'),
    MTOB  = dict(rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'),
    TEC   = dict(matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
                 rphiRecHits    = 'siStripMatchedRecHitsTop:rphiRecHit')
)

simpleCosmicBONSeedsTop = simpleCosmicBONSeeds.clone(
    PositiveYOnly    = True,
    NegativeYOnly    = False,
    ClusterCheckPSet = dict(
	ClusterCollectionLabel    = 'siStripClustersTop',
	MaxNumberOfStripClusters = 150,
    ),
    TripletsSrc      = 'simpleCosmicBONSeedingLayersTop',
)

combinedP5SeedsForCTFTop = globalCombinedSeeds.clone(
    seedCollections = ['combinatorialcosmicseedfinderP5Top',
                       'simpleCosmicBONSeedsTop']
)

MeasurementTrackerTop = MeasurementTracker.clone(
    ComponentName = 'MeasurementTrackerTop'
)

GroupedCkfTrajectoryBuilderP5Top = GroupedCkfTrajectoryBuilderP5.clone()

ckfTrackCandidatesP5Top = ckfTrackCandidatesP5.clone(
    TrajectoryBuilderPSet = dict(refToPSet_ = 'GroupedCkfTrajectoryBuilderP5Top'),
    NavigationSchool      = 'CosmicNavigationSchool',
    src                   = 'combinedP5SeedsForCTFTop', #ok for 32X
    useHitsSplitting      = True
)

ctfWithMaterialTracksP5Top = ctfWithMaterialTracksCosmics.clone(
    src                = 'ckfTrackCandidatesP5Top',
    Fitter             = 'FittingSmootherRKP5',
    clusterRemovalInfo = 'topBottomClusterInfoProducerTop'
)

ctftracksP5TopTask = cms.Task(combinatorialcosmicseedingtripletsP5Top,
                              combinatorialcosmicseedingpairsTOBP5Top,
                              combinatorialcosmicseedingpairsTECposP5Top,
                              combinatorialcosmicseedingpairsTECnegP5Top,
                              combinatorialcosmicseedfinderP5Top,
                              simpleCosmicBONSeedingLayersTop,
                              simpleCosmicBONSeedsTop,
                              combinedP5SeedsForCTFTop,
                              ckfTrackCandidatesP5Top,
                              ctfWithMaterialTracksP5Top)
ctftracksP5Top = cms.Sequence(ctftracksP5TopTask)

###CKF BOTTOM
combinatorialcosmicseedingtripletsP5Bottom = combinatorialcosmicseedingtripletsP5.clone(
    TIB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTIB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TOB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTOB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TEC  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
                rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit')
)

combinatorialcosmicseedingpairsTOBP5Bottom = combinatorialcosmicseedingpairsTOBP5.clone(
    TIB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTIB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TOB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTOB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TEC  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
                rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit')
)

combinatorialcosmicseedingpairsTECposP5Bottom = combinatorialcosmicseedingpairsTECposP5.clone(
    TEC = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
               rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit')
)

combinatorialcosmicseedingpairsTECnegP5Bottom = combinatorialcosmicseedingpairsTECnegP5.clone(
    TEC = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
               rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit')
)

combinatorialcosmicseedfinderP5Bottom = combinatorialcosmicseedfinderP5.clone(
    SeedsFromPositiveY      = False,
    SeedsFromNegativeY      = True,
    ClusterCollectionLabel    = 'siStripClustersBottom',
    MaxNumberOfStripClusters = 150,
    OrderedHitsFactoryPSets = {0: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingtripletsP5Bottom"),
                               1: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingpairsTOBP5Bottom"),
                               2: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingpairsTECposP5Bottom"),
                               3: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingpairsTECposP5Bottom"),
                               4: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Bottom"),
                               5: dict(PropagationDirection = 'oppositeToMomentum', LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Bottom"),
    }
)

simpleCosmicBONSeedingLayersBottom = simpleCosmicBONSeedingLayers.clone(
    TIB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTIB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TOB  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit'),
    MTOB = dict(rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'),
    TEC  = dict(matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
                rphiRecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit')
)

simpleCosmicBONSeedsBottom = simpleCosmicBONSeeds.clone(
    PositiveYOnly    = False,
    NegativeYOnly    = True,
    ClusterCheckPSet = dict(
	ClusterCollectionLabel    = 'siStripClustersBottom',
        MaxNumberOfStripClusters = 150
    ),
    TripletsSrc      = 'simpleCosmicBONSeedingLayersBottom'
)

combinedP5SeedsForCTFBottom = globalCombinedSeeds.clone(
    seedCollections = ['combinatorialcosmicseedfinderP5Bottom',
                       'simpleCosmicBONSeedsBottom']
)

MeasurementTrackerBottom = MeasurementTracker.clone(
    ComponentName = 'MeasurementTrackerBottom'
)

GroupedCkfTrajectoryBuilderP5Bottom = GroupedCkfTrajectoryBuilderP5.clone()

ckfTrackCandidatesP5Bottom = ckfTrackCandidatesP5.clone(
    TrajectoryBuilderPSet = dict(refToPSet_ = 'GroupedCkfTrajectoryBuilderP5Bottom'),
    NavigationSchool      = 'CosmicNavigationSchool',
    src                   = 'combinedP5SeedsForCTFBottom', #ok for 32X
    useHitsSplitting = True
)

ctfWithMaterialTracksP5Bottom = ctfWithMaterialTracksCosmics.clone(
    src                = 'ckfTrackCandidatesP5Bottom',
    Fitter             = 'FittingSmootherRKP5',
    clusterRemovalInfo = 'topBottomClusterInfoProducerBottom'
)

ctftracksP5BottomTask = cms.Task(combinatorialcosmicseedingtripletsP5Bottom,
                                 combinatorialcosmicseedingpairsTOBP5Bottom,
                                 combinatorialcosmicseedingpairsTECposP5Bottom,
                                 combinatorialcosmicseedingpairsTECnegP5Bottom,
                                 combinatorialcosmicseedfinderP5Bottom,
                                 simpleCosmicBONSeedingLayersBottom,
                                 simpleCosmicBONSeedsBottom,
                                 combinedP5SeedsForCTFBottom,
                                 ckfTrackCandidatesP5Bottom,
                                 ctfWithMaterialTracksP5Bottom)
ctftracksP5Bottom = cms.Sequence(ctftracksP5BottomTask)

#COSMIC TOP
cosmicseedfinderP5Top = cosmicseedfinderP5.clone(
    stereorecHits             = 'siStripMatchedRecHitsTop:stereoRecHit',
    matchedRecHits            = 'siStripMatchedRecHitsTop:matchedRecHit',
    rphirecHits               = 'siStripMatchedRecHitsTop:rphiRecHit',
    ClusterCollectionLabel    = 'siStripClustersTop',
    PositiveYOnly             = True,
    NegativeYOnly             = False,
    MaxNumberOfStripClusters = 150
)

cosmicCandidateFinderP5Top = cosmicCandidateFinderP5.clone(
    cosmicSeeds    = 'cosmicseedfinderP5Top',
    stereorecHits  = 'siStripMatchedRecHitsTop:stereoRecHit',
    HitProducer    = 'siStripRecHitsTop',
    pixelRecHits   = 'siPixelRecHitsTop',
    matchedRecHits = 'siStripMatchedRecHitsTop:matchedRecHit',
    rphirecHits    = 'siStripMatchedRecHitsTop:rphiRecHit'
)

cosmictrackfinderP5Top = cosmictrackfinderCosmics.clone(
    src                = 'cosmicCandidateFinderP5Top',
    clusterRemovalInfo = 'topBottomClusterInfoProducerTop'
)

cosmictracksP5TopTask = cms.Task(cosmicseedfinderP5Top,
                                 cosmicCandidateFinderP5Top,
                                 cosmictrackfinderP5Top)
cosmictracksP5Top = cms.Sequence(cosmictracksP5TopTask)

#COSMIC BOTTOM
cosmicseedfinderP5Bottom = cosmicseedfinderP5.clone(
    stereorecHits             = 'siStripMatchedRecHitsBottom:stereoRecHit',
    matchedRecHits            = 'siStripMatchedRecHitsBottom:matchedRecHit',
    rphirecHits               = 'siStripMatchedRecHitsBottom:rphiRecHit',
    ClusterCollectionLabel    = 'siStripClustersBottom',
    PositiveYOnly             = False,
    NegativeYOnly             = True,
    MaxNumberOfStripClusters = 150
)

cosmicCandidateFinderP5Bottom = cosmicCandidateFinderP5.clone(
    cosmicSeeds    = 'cosmicseedfinderP5Bottom',
    stereorecHits  = 'siStripMatchedRecHitsBottom:stereoRecHit',
    HitProducer    = 'siStripRecHitsBottom',
    pixelRecHits   = 'siPixelRecHitsBottom',
    matchedRecHits = 'siStripMatchedRecHitsBottom:matchedRecHit',
    rphirecHits    = 'siStripMatchedRecHitsBottom:rphiRecHit'
)

cosmictrackfinderP5Bottom = cosmictrackfinderCosmics.clone(
    src                = 'cosmicCandidateFinderP5Bottom',
    clusterRemovalInfo = 'topBottomClusterInfoProducerBottom'
)

cosmictracksP5BottomTask = cms.Task(cosmicseedfinderP5Bottom,
                                    cosmicCandidateFinderP5Bottom,
                                    cosmictrackfinderP5Bottom)
cosmictracksP5Bottom = cms.Sequence(cosmictracksP5BottomTask)

#TOP SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
tracksP5TopTask = cms.Task(ctftracksP5TopTask, cosmictracksP5TopTask)
tracksP5Top = cms.Sequence(tracksP5TopTask)
#BOTTOM SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
tracksP5BottomTask = cms.Task(ctftracksP5BottomTask, cosmictracksP5BottomTask)
tracksP5Bottom = cms.Sequence(tracksP5BottomTask)
