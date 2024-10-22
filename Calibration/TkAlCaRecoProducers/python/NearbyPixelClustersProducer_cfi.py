import FWCore.ParameterSet.Config as cms

NearbyPixelClustersProducer = cms.EDProducer('NearbyPixelClustersProducer',
                                             clusterCollection = cms.InputTag('siPixelClusters'), # input clusters
                                             trajectoryInput = cms.InputTag('TrackerRefitter'),   # input trajectories
                                             throwBadComponents = cms.bool(False),                # do not use bad components
                                             dumpWholeDetIds = cms.bool(False)                    # write all clusters in Det
                                             )
