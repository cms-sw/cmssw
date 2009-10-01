import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.Reconstruction_cff import *
from CalibTracker.SiStripCommon.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefitRAW"
shallowTrackClusters.Clusters = 'CalibrationTracksRAW'
shallowClusters.Clusters = 'CalibrationTracksRAW'

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitRAW + ntuple )
schedule = cms.Schedule( reconstruction_step, filter_refit_ntuplize_step )
