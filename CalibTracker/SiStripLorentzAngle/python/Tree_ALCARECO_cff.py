import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefit"
shallowTrackClusters.Clusters = 'CalibrationTracks'
shallowClusters.Clusters = 'CalibrationTracks'

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefit + ntuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
