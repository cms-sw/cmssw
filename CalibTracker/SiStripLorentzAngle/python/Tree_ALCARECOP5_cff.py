import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefitAlcaP5"
shallowTrackClusters.Clusters = 'CalibrationTracksAlcaP5'
shallowClusters.Clusters = 'CalibrationTracksAlcaP5'

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitAlcaP5 + LorentzAngleNtuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
