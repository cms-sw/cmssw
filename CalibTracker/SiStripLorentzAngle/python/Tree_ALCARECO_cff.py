import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefitAlca"
shallowTrackClusters.Clusters = 'CalibrationTracksAlca'
shallowClusters.Clusters = 'CalibrationTracksAlca'

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitAlca + LorentzAngleNtuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
