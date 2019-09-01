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
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(clusterSummaryProducer,
  doStrips = False,
  stripClusters = ''
)
clusterSummaryProducerNoSplitting = clusterSummaryProducer.clone(pixelClusters = 'siPixelClusters')
