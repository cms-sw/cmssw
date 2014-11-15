import FWCore.ParameterSet.Config as cms

DefaultClusterizer = cms.PSet(
    Algorithm = cms.string('ThreeThresholdAlgorithm'),
    ChannelThreshold = cms.double(2.0),
    SeedThreshold = cms.double(3.0),
    ClusterThreshold = cms.double(5.0),
    MaxSequentialHoles = cms.uint32(0),
    MaxSequentialBad = cms.uint32(1),
    MaxAdjacentBad = cms.uint32(0),
    QualityLabel = cms.string(""),
    RemoveApvShots     = cms.bool(True),
    minGoodCharge = cms.double(-2069),
    doRefineCluster = cms.bool(False),
    occupancyThreshold = cms.double(0.05),
    widthThreshold = cms.uint32(4)
    )
