import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_cff import *
iterHLTTracksMonitoringHLT.doProfilesVsLS   = cms.bool(True)
iterHLTTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")
pixelTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

pixelTracksForHighBetaMonitoringHLT = pixelTracksMonitoringHLT.clone()
pixelTracksForHighBetaMonitoringHLT.FolderName = "HLT/Tracking/pixelTracksForHighBeta"
pixelTracksForHighBetaMonitoringHLT.TrackProducer = "hltPixelTracksForHighBeta"
pixelTracksForHighBetaMonitoringHLT.allTrackProducer = "hltPixelTracksForHighBeta"

pixelTracksForHighBetaBPixMonitoringHLT = pixelTracksMonitoringHLT.clone()
pixelTracksForHighBetaBPixMonitoringHLT.FolderName = "HLT/Tracking/pixelTracksForHighBetaBPix"
pixelTracksForHighBetaBPixMonitoringHLT.TrackProducer = "hltPixelTracksForHighBetaBPix"
pixelTracksForHighBetaBPixMonitoringHLT.allTrackProducer = "hltPixelTracksForHighBetaBPix"


trackingMonitoringHLTsequence = cms.Sequence(
    pixelTracksMonitoringHLT # hltPixel tracks monitoring
    * pixelTracksForHighBetaMonitoringHLT # hltPixelTrackHighBeta monitoring for 90m beta star
    * pixelTracksForHighBetaBPixMonitoringHLT # hltPixelTrackHighBetaBPix monitoring for 90m beta star
    * iter2MergedTracksMonitoringHLT # hltIter2Merged tracks monitoring    
    * iterHLTTracksMonitoringHLT # hltTracksMerged tracks monitoring
)

egmTrackingMonitorHLTsequence = cms.Sequence(
    gsfTracksMonitoringHLT
    * pixelTracksForElectronsTracksMonitoringHLT
    * iterHLTTracksForElectronsMonitoringHLT
)
