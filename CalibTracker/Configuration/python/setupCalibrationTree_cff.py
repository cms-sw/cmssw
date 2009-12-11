import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
from CalibTracker.SiStripChannelGain.ntuple_cff import *
from CalibTracker.SiStripHitEfficiency.SiStripHitEff_cff import *
from CalibTracker.SiStripCommon.theBigNtuple_cfi import *

shallowTrackClusters.Tracks = "CalibrationTracksRefit"
shallowTrackClusters.Clusters = 'CalibrationTracks'
shallowClusters.Clusters = 'CalibrationTracks'
shallowGainCalibration.Tracks = "CalibrationTracksRefit"
anEff.combinatorialTracks = "CalibrationTracksRefit"
anEff.trajectories = "CalibrationTracksRefit"

#Schedule
filter_refit_ntuplize_step = cms.Path( siStripMatchedRecHits + trackFilterRefit + theBigNtuple + LorentzAngleNtuple + hiteff + OfflineGainNtuple)
schedule = cms.Schedule( filter_refit_ntuplize_step )






