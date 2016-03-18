import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
                                                  ClusterCheckPSet = cms.PSet(
                                                      PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
                                                      MaxNumberOfCosmicClusters = cms.uint32(400000),
                                                      doClusterCheck = cms.bool(True),
                                                      ClusterCollectionLabel = cms.InputTag("siStripClusters"),
                                                      MaxNumberOfPixelClusters = cms.uint32(40000),
                                                      cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)"),
                                                      ),
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
eras.trackingPhase1.toModify(photonConvTrajSeedFromSingleLeg,
    ClusterCheckPSet = dict(
        MaxNumberOfCosmicClusters = 1000000,
        MaxNumberOfPixelClusters = 100000,
        cut = 'strip < 1000000 && pixel < 100000 && (strip < 100000 + 20*pixel) && (pixel < 20000 + 0.1*strip)'
    ),
    OrderedHitsFactoryPSet = dict(maxElement = 100000),
    RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.3)),
)
