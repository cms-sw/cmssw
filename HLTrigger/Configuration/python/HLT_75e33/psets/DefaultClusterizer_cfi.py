import FWCore.ParameterSet.Config as cms

DefaultClusterizer = cms.PSet(
    Algorithm = cms.string('ThreeThresholdAlgorithm'),
    ChannelThreshold = cms.double(2.0),
    ClusterThreshold = cms.double(5.0),
    MaxAdjacentBad = cms.uint32(0),
    MaxSequentialBad = cms.uint32(1),
    MaxSequentialHoles = cms.uint32(0),
    QualityLabel = cms.string(''),
    RemoveApvShots = cms.bool(True),
    SeedThreshold = cms.double(3.0),
    clusterChargeCut = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutNone')
    )
)