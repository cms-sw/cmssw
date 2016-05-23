import FWCore.ParameterSet.Config as cms

from CalibTracker.Configuration.Filter_Refit_cff import *
from CalibTracker.SiStripLorentzAngle.ntuple_cff import *
from CalibTracker.SiStripChannelGain.ntuple_cff import *
from CalibTracker.SiStripHitEfficiency.SiStripHitEff_cff import *

shallowTrackClusters.Tracks             = 'CalibrationTracksRefit'
shallowTrackClusters.Clusters           = 'CalibrationTracks'
shallowClusters.Clusters                = 'CalibrationTracks'
shallowGainCalibration.Tracks           = 'CalibrationTracksRefit'
anEff.combinatorialTracks               = 'CalibrationTracksRefit'
anEff.trajectories                      = 'CalibrationTracksRefit'

#Schedule
#TkCalFullSequence = cms.Sequence( trackFilterRefit + LorentzAngleNtuple + hiteff + OfflineGainNtuple)
TkCalSeq_StdBunch   = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_StdBunch + hiteff)
TkCalSeq_StdBunch0T = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_StdBunch0T + hiteff)
TkCalSeq_IsoMuon   = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_IsoMuon + hiteff)
TkCalSeq_IsoMuon0T = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_IsoMuon0T + hiteff)
TkCalSeq_AagBunch   = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_AagBunch + hiteff)
TkCalSeq_AagBunch0T = cms.Sequence(MeasurementTrackerEvent + trackFilterRefit + OfflineGainNtuple_AagBunch0T + hiteff)


