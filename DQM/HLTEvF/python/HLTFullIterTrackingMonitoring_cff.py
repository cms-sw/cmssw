import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TrackingMonitoring_cff import *

fullIterTracksMonitoringHLT = trackingMonHLT.clone()
fullIterTracksMonitoringHLT.FolderName       = 'HLT/Tracking/FullIterativeTrackingMergedForRefPP'
fullIterTracksMonitoringHLT.TrackProducer    = 'hltFullIterativeTrackingMergedForRefPP'
fullIterTracksMonitoringHLT.allTrackProducer = 'hltFullIterativeTrackingMergedForRefPP'

