import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_cff import *
iterHLTTracksMonitoringHLT.doProfilesVsLS   = cms.bool(True)
iterHLTTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")
pixelTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

trackingMonitoringHLTsequence = cms.Sequence(
    pixelTracksMonitoringHLT # hltPixel tracks monitoring
    * iter2MergedTracksMonitoringHLT # hltIter2Merged tracks monitoring    
    * iterHLTTracksMonitoringHLT # hltTracksMerged tracks monitoring
)

egmTrackingMonitorHLTsequence = cms.Sequence(
    gsfTracksMonitoringHLT
    * pixelTracksForElectronsTracksMonitoringHLT
    * iterHLTTracksForElectronsMonitoringHLT
)
