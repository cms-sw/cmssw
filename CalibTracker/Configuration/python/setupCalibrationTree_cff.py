import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
from CalibTracker.SiStripChannelGain.ntuple_cff import *
shallowTrackClusters.Tracks = "CalibrationTracksRefit"
shallowTrackClusters.Clusters = 'CalibrationTracks'
shallowClusters.Clusters = 'CalibrationTracks'

from CalibTracker.SiStripHitEfficiency.SiStripHitEff_cff import *

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefit + ntuple + hiteff + OfflineGainNtuple)
schedule = cms.Schedule( filter_refit_ntuplize_step )






