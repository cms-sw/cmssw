import FWCore.ParameterSet.Config as cms

photonConvTrajSeedFromQuadruplets  = cms.EDProducer("PhotonConversionTrajectorySeedProducerFromQuadruplets",
                                                  TrackRefitter        = cms.InputTag('TrackRefitter',''),
                                                  primaryVerticesTag   = cms.InputTag("offlinePrimaryVertices"), 
                                                  beamSpotInputTag     = cms.InputTag("offlineBeamSpot"),
                                                  newSeedCandidates    = cms.string("conv2SeedCandidates"),
                                                  xcheckSeedCandidates = cms.string("xcheckSeedCandidates"),
                                                  DoxcheckSeedCandidates = cms.bool(False),
                                                  OrderedHitsFactoryPSet = cms.PSet(
                                                      maxElement = cms.uint32(900000),
                                                      SeedingLayers = cms.InputTag('conv2LayerPairs')
                                                      ),
                                                  SeedComparitorPSet = cms.PSet(
                                                      ComponentName = cms.string('PixelClusterShapeSeedComparitor'), #'LowPtClusterShapeSeedComparitor') #none
                                                      FilterAtHelixStage = cms.bool(True),#Def: True
                                                      FilterPixelHits = cms.bool(False),#Def: False
                                                      FilterStripHits = cms.bool(True),#Def: True                                                      
                                                      ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
                                                      ),
                                                  ClusterCheckPSet = cms.PSet(
                                                      PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
                                                      MaxNumberOfCosmicClusters = cms.uint32(50000),
                                                      doClusterCheck = cms.bool(True),
                                                      ClusterCollectionLabel = cms.InputTag("siStripClusters"),
                                                      MaxNumberOfPixelClusters = cms.uint32(10000)
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
                                                      ComponentName = cms.string('SeedForPhotonConversionFromQuadruplets'),
                                                      SeedMomentumForBOFF = cms.double(5.0),
                                                      propagator = cms.string('PropagatorWithMaterial')
                                                      ),
                                                  QuadCutPSet = cms.PSet(
                                                     Cut_minLegPt = cms.double(0.6), #GeV
                                                     Cut_maxLegPt = cms.double(10.), #GeV
                                                     rejectAllQuads = cms.bool(False),
                                                     apply_DeltaPhiCuts = cms.bool(True),
                                                     apply_Arbitration = cms.bool(True),
                                                     apply_ClusterShapeFilter = cms.bool(True),
                                                     apply_zCACut = cms.bool(False),
                                                     Cut_zCA = cms.double(100), #cm
                                                     Cut_DeltaRho = cms.double(12.), #cm
                                                     Cut_BeamPipeRadius = cms.double(3.) #cm
                                                      )
                                                  )
                                                  
                                                  
                                                  
                                                  
