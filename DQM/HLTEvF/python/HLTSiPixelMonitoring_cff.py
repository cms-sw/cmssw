import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_cff import *

hltSiPixelPhase1ClustersAnalyzer.pixelSrc = cms.InputTag("hltSiPixelClustersForHighBeta")
hltSiPixelPhase1TrackClustersAnalyzer.clusters = cms.InputTag("hltSiPixelClustersForHighBeta")
hltSiPixelPhase1TrackClustersAnalyzer.tracks = cms.InputTag("hltPixelTracksForHighBeta")

hltSiPixelPhase1TrackClustersOnTrackNClusters.range_max = 50
hltSiPixelPhase1TrackClustersOnTrackNClusters.range_nbins = 50

hltSiPixelPhase1ClustersNClusters.range_max = 50
hltSiPixelPhase1ClustersNClusters.range_nbins = 50

hltSiPixelPhase1ClustersNClustersInclusive.range_max = 50
hltSiPixelPhase1ClustersNClustersInclusive.range_nbins = 50

hltSiPixelPhase1ClustersReadoutNClusters.range_max = 50
hltSiPixelPhase1ClustersReadoutNClusters.range_nbins = 50


pixelOnlineMonitorHLTsequence = cms.Sequence(
    sipixelMonitorHLTsequence
)
