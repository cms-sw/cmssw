import FWCore.ParameterSet.Config as cms

clusterSummaryProducer = cms.EDProducer('ClusterSummaryProducer',
                                        stripClusters=cms.InputTag('siStripClusters'),
                                        pixelClusters=cms.InputTag('siPixelClustersPreSplitting'),
                                        doStrips=cms.bool(True),
                                        doPixels=cms.bool(True),
                                        verbose=cms.bool(False),
                                        wantedSubDets = cms.vstring("TOB","TIB","TID","TEC","STRIP","BPIX","FPIX","PIXEL"),
                                        wantedUserSubDets = cms.VPSet()
                                        )
clusterSummaryProducerNoSplitting = clusterSummaryProducer.clone(pixelClusters = 'siPixelClusters')
