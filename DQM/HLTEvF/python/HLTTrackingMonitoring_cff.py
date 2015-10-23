import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_cff import *
iterHLTTracksMonitoringHLT.doProfilesVsLS   = cms.bool(True)
iterHLTTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")
pixelTracksMonitoringHLT.beamSpot = cms.InputTag("hltOnlineBeamSpot")

from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

trackingMonitoringHLTsequence = cms.Sequence(
    pixelTracksMonitoringHLT # hltPixel tracks monitoring
    * iterHLTTracksMonitoringHLT # hltIter2Merged tracks monitoring
)
