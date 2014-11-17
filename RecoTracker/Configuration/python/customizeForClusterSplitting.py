import FWCore.ParameterSet.Config as cms

def customizeForClusterSplitting(process):
     process.load('RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi')	
     clustersTmp = 'siPixelClustersPreSplitting'
     # 0. Produce tmp clusters in the first place.
     process.siPixelClustersPreSplitting = process.siPixelClusters.clone()
     process.siPixelRecHitsPreSplitting = process.siPixelRecHits.clone()
     process.siPixelRecHitsPreSplitting.src = clustersTmp
     process.pixeltrackerlocalreco.replace(process.siPixelClusters, process.siPixelClustersPreSplitting)
     process.pixeltrackerlocalreco.replace(process.siPixelRecHits, process.siPixelRecHitsPreSplitting)
     process.clusterSummaryProducer.pixelClusters = clustersTmp
 
     # 0.5 Feed them to MTE and PixelClusterShapeCache
 
     process.MeasurementTrackerEventPreSplitting = process.MeasurementTrackerEvent.clone()
     process.MeasurementTrackerEventPreSplitting.pixelClusterProducer = clustersTmp
     process.siPixelClusterShapeCachePreSplitting = process.siPixelClusterShapeCache.clone()
     process.siPixelClusterShapeCachePreSplitting.src = clustersTmp
     process.globalreco.replace(process.MeasurementTrackerEvent, process.MeasurementTrackerEventPreSplitting)
     process.globalreco.replace(process.siPixelClusterShapeCache, process.siPixelClusterShapeCachePreSplitting)
 
     # 1. clone what needs to be cloned to have PV before Cluster
     # Splitting, put together the sequence and prepend it to the main
     # iterative sequence
     process.initialStepSeedLayersPreSplitting = process.initialStepSeedLayers.clone()
     process.initialStepSeedLayersPreSplitting.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
     process.initialStepSeedLayersPreSplitting.BPix.HitProducer = 'siPixelRecHitsPreSplitting'
 
     process.initialStepSeedsPreSplitting = process.initialStepSeeds.clone()
     process.initialStepSeedsPreSplitting.OrderedHitsFactoryPSet.SeedingLayers = 'initialStepSeedLayersPreSplitting'
     process.initialStepSeedsPreSplitting.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'
     process.initialStepSeedsPreSplitting.ClusterCheckPSet.PixelClusterCollectionLabel = clustersTmp
 
     process.initialStepTrackCandidatesPreSplitting = process.initialStepTrackCandidates.clone()
     process.initialStepTrackCandidatesPreSplitting.src = 'initialStepSeedsPreSplitting'
     process.initialStepTrackCandidatesPreSplitting.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
 
     process.initialStepTracksPreSplitting = process.initialStepTracks.clone()
     process.initialStepTracksPreSplitting.src = 'initialStepTrackCandidatesPreSplitting'
     process.initialStepTracksPreSplitting.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'
 
     process.firstStepPrimaryVerticesPreSplitting = process.firstStepPrimaryVertices.clone()
     process.firstStepPrimaryVerticesPreSplitting.TrackLabel = 'initialStepTracksPreSplitting'
 
     process.initialStepTrackRefsForJetsPreSplitting = process.initialStepTrackRefsForJets.clone()
     process.initialStepTrackRefsForJetsPreSplitting.src = 'initialStepTracksPreSplitting'
     process.caloTowerForTrkPreSplitting = process.caloTowerForTrk.clone()
     process.ak4CaloJetsForTrkPreSplitting = process.ak4CaloJetsForTrk.clone()
     process.ak4CaloJetsForTrkPreSplitting.src = 'caloTowerForTrkPreSplitting'
     process.ak4CaloJetsForTrkPreSplitting.srcPVs = 'firstStepPrimaryVerticesPreSplitting'
     process.jetsForCoreTrackingPreSplitting = process.jetsForCoreTracking.clone()
     process.jetsForCoreTrackingPreSplitting.src = 'ak4CaloJetsForTrkPreSplitting'
 
     process.siPixelClusters = process.jetCoreClusterSplitter.clone(
          pixelClusters         = cms.InputTag(clustersTmp),
          vertices              = cms.InputTag('firstStepPrimaryVerticesPreSplitting'),
          cores = cms.InputTag("jetsForCoreTrackingPreSplitting"),
     )
 
 
     process.InitialStepPreSplitting = cms.Sequence(process.initialStepSeedLayersPreSplitting +
                                                    process.initialStepSeedsPreSplitting +
                                                    process.initialStepTrackCandidatesPreSplitting +
                                                    process.initialStepTracksPreSplitting +
                                                    process.firstStepPrimaryVerticesPreSplitting +
                                                    process.initialStepTrackRefsForJetsPreSplitting +
                                                    process.caloTowerForTrkPreSplitting +
                                                    process.ak4CaloJetsForTrkPreSplitting +
                                                    process.jetsForCoreTrackingPreSplitting +
                                                    process.siPixelClusters +
                                                    process.siPixelRecHits +
                                                    process.MeasurementTrackerEvent +
                                                    process.siPixelClusterShapeCache)
 
     process.iterTracking.insert(0,process.InitialStepPreSplitting)
 
     return process
