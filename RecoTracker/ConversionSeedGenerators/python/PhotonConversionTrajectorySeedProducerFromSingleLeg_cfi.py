import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi import seedGeneratorFromRegionHitsEDProducer 
CommonClusterCheckPSet = seedGeneratorFromRegionHitsEDProducer.ClusterCheckPSet


photonConvTrajSeedFromSingleLeg  = cms.EDProducer("PhotonConversionTrajectorySeedProducerFromSingleLeg",
                                                  TrackRefitter        = cms.InputTag('TrackRefitter',''),
                                                  primaryVerticesTag   = cms.InputTag("offlinePrimaryVertices"),
                                                  beamSpotInputTag     = cms.InputTag("offlineBeamSpot"),
                                                  newSeedCandidates    = cms.string("convSeedCandidates"),
                                                  xcheckSeedCandidates = cms.string("xcheckSeedCandidates"),
                                                  vtxMinDoF            = cms.double(4),
                                                  maxDZSigmas          = cms.double(10.),
                                                  maxNumSelVtx         = cms.uint32(2),
                                                  applyTkVtxConstraint = cms.bool(True),
                                                  
                                                  DoxcheckSeedCandidates = cms.bool(False),
                                                  OrderedHitsFactoryPSet = cms.PSet(
                                                      maxHitPairsPerTrackAndGenerator = cms.uint32(10),
                                                      maxElement = cms.uint32(40000),
                                                      SeedingLayers = cms.InputTag('convLayerPairs')
                                                      ),
                                                  SeedComparitorPSet = cms.PSet(
                                                      ComponentName = cms.string('none')
                                                      ),
                                                  ClusterCheckPSet = CommonClusterCheckPSet,
                                                  RegionFactoryPSet = cms.PSet(
                                                      RegionPSet = cms.PSet( precise = cms.bool(True),
                                                                             beamSpot = cms.InputTag("offlineBeamSpot"),
                                                                             originRadius = cms.double(3.0),
                                                                             ptMin = cms.double(0.2),
                                                                             originHalfLength = cms.double(12.0)
                                                                             ),
                                                      ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
                                                      ),
                                                  SeedCreatorPSet = cms.PSet(
                                                      ComponentName = cms.string('SeedForPhotonConversion1Leg'),
                                                      SeedMomentumForBOFF = cms.double(5.0),
                                                      propagator = cms.string('PropagatorWithMaterial'),
                                                      TTRHBuilder = cms.string('WithTrackAngle')
                                                      )
                                                  )
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(photonConvTrajSeedFromSingleLeg,
    OrderedHitsFactoryPSet = dict(maxElement = 10000),
    ClusterCheckPSet = dict(
        MaxNumberOfCosmicClusters = 150000,
        MaxNumberOfPixelClusters = 20000,
        cut = "strip < 150000 && pixel < 20000 && (strip < 20000 + 7* pixel)"
    )
)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(photonConvTrajSeedFromSingleLeg,
    ClusterCheckPSet = dict(
        MaxNumberOfCosmicClusters = 1000000,
        MaxNumberOfPixelClusters = 100000,
        cut = None
    ),
    OrderedHitsFactoryPSet = dict(maxElement = 100000),
    RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.3)),
)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
peripheralPbPb.toModify(photonConvTrajSeedFromSingleLeg,
                        ClusterCheckPSet = dict(cut = "strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA ).toModify(photonConvTrajSeedFromSingleLeg,
                             ClusterCheckPSet = dict(MaxNumberOfPixelClusters = 100000,
                                                     cut = "strip < 1000000 && pixel < 100000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/2.)"
                                                     ),
                             OrderedHitsFactoryPSet = dict(maxElement = 100000)
)
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
(pp_on_XeXe_2017 | pp_on_AA ).toModify(photonConvTrajSeedFromSingleLeg,
               RegionFactoryPSet = dict(ComponentName = 'GlobalTrackingRegionWithVerticesProducer',
                                        RegionPSet = _globalTrackingRegionWithVertices.RegionPSet.clone(
                                                          originRadius = 0,
                                                          originRScaling4BigEvts = True,
                                                          minOriginR = 0,
                                                          scalingStartNPix = 0,
                                                          scalingEndNPix = 1#essentially turn off immediately 
                                                         ),
                                        )
)
