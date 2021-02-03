import FWCore.ParameterSet.Config as cms

clusterSummaryProducer = cms.EDProducer("ClusterSummaryProducer",
    doPixels = cms.bool(True),
    doStrips = cms.bool(False),
    pixelClusters = cms.InputTag("siPixelClustersPreSplitting"),
    stripClusters = cms.InputTag(""),
    verbose = cms.bool(False),
    wantedSubDets = cms.vstring(
        'TOB',
        'TIB',
        'TID',
        'TEC',
        'STRIP',
        'BPIX',
        'FPIX',
        'PIXEL'
    ),
    wantedUserSubDets = cms.VPSet()
)
