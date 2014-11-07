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
    doRefineCluster = cms.bool(True),
    occupancyThreshold = cms.double(0.08),
    useMCtruth = cms.bool(True),
#  For TrackerHitAssociator
#  If pileup and crossing frames available:
    # ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof',
    #                       'g4SimHitsTrackerHitsTIBHighTof',
    #                       'g4SimHitsTrackerHitsTIDLowTof',
    #                       'g4SimHitsTrackerHitsTIDHighTof',
    #                       'g4SimHitsTrackerHitsTOBLowTof',
    #                       'g4SimHitsTrackerHitsTOBHighTof',
    #                       'g4SimHitsTrackerHitsTECLowTof',
    #                       'g4SimHitsTrackerHitsTECHighTof'),
#  otherwise:
    ROUList = cms.vstring('TrackerHitsTIBLowTof',
                          'TrackerHitsTIBHighTof',
                          'TrackerHitsTIDLowTof',
                          'TrackerHitsTIDHighTof',
                          'TrackerHitsTOBLowTof',
                          'TrackerHitsTOBHighTof',
                          'TrackerHitsTECLowTof',
                          'TrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(True),
    associatePixel = cms.bool(False),
    associateStrip = cms.bool(True)
)
