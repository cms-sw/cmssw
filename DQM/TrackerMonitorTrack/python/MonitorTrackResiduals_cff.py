import FWCore.ParameterSet.Config as cms

#TrackRefitter With Material
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitter.TrajectoryInEvent = True
# usually without refit: # TransientTrackingRecHitBuilder: no refit of hits...
#TrackRefitter.TTRHBuilder = 'WithoutRefit'
#from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
# ... but matching for strip stereo should be redone: 
#ttrhbwor.Matcher = 'StandardMatcher'

import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsTier0 = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsTier0.OutputMEsInRootFile = False
MonitorTrackResidualsTier0.Mod_On = False

import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsDQM = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsDQM.OutputMEsInRootFile = False
MonitorTrackResidualsDQM.Mod_On = True

import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsStandAlone = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsStandAlone.OutputMEsInRootFile = True
MonitorTrackResidualsStandAlone.Mod_On = False

# Sequences
DQMMonitorTrackResidualsTier0 = cms.Sequence(TrackRefitter*MonitorTrackResidualsTier0)
DQMMonitorTrackResiduals = cms.Sequence(TrackRefitter*MonitorTrackResidualsDQM)
DQMMonitorTrackResidualsStandAlone = cms.Sequence(TrackRefitter*MonitorTrackResidualsStandAlone)


