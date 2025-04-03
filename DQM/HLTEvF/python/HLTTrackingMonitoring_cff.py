import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_cff import *
iterHLTTracksMonitoringHLT.doProfilesVsLS = True
iterHLTTracksMonitoringHLT.beamSpot = "hltOnlineBeamSpot"
pixelTracksMonitoringHLT.beamSpot = "hltOnlineBeamSpot"

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

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(trackingMonitoringHLTsequence, cms.Sequence(pixelTracksMonitoringHLT * iterHLTTracksMonitoringHLT * doubletRecoveryHPTracksMonitoringHLT)) # * iter0HPTracksMonitoringHLT ))
run3_common.toReplaceWith(egmTrackingMonitorHLTsequence, cms.Sequence(gsfTracksMonitoringHLT))
