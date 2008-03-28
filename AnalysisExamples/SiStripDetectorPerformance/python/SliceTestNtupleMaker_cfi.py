import FWCore.ParameterSet.Config as cms

# SliceTestNtupleMaker Module default configuration
modSliceTestNtupleMaker = cms.EDFilter("SliceTestNtupleMaker",
    oClusterInfo = cms.untracked.InputTag("siStripClusterInfoProducer"),
    # Output ROOT file name
    oOFileName = cms.untracked.string('SliceTestNtupleMaker_out.root'),
    oTrack = cms.untracked.InputTag("cosmictrackfinder"),
    # TrackInfos Labels
    oDigi = cms.untracked.InputTag("siStripZeroSuppression","VirginRaw"),
    oTTRHBuilder = cms.untracked.string('WithTrackAngle'),
    oCluster = cms.untracked.InputTag("siStripClusters")
)


