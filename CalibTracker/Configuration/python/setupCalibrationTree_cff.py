import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
from CalibTracker.SiStripChannelGain.ntuple_cff import *
from CalibTracker.SiStripHitEfficiency.SiStripHitEff_cff import *

shallowTrackClusters.Tracks = "CalibrationTracksRefit"
shallowTrackClusters.Clusters = 'CalibrationTracks'
shallowClusters.Clusters = 'CalibrationTracks'
shallowGainCalibration.Tracks = "CalibrationTracksRefit"
anEff.combinatorialTracks = "CalibrationTracksRefit"
anEff.trajectories = "CalibrationTracksRefit"

#Schedule
#TkCalFullSequence = cms.Sequence( trackFilterRefit + LorentzAngleNtuple + hiteff + OfflineGainNtuple)
TkCalFullSequence = cms.Sequence( trackFilterRefit + OfflineGainNtuple + hiteff)
