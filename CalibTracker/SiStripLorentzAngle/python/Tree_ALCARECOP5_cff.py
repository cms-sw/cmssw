import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefitP5"
shallowTrackClusters.Clusters = 'CalibrationTracksP5'
shallowClusters.Clusters = 'CalibrationTracksP5'

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitP5 + ntuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
