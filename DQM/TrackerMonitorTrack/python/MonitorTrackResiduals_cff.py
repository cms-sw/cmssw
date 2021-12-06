import FWCore.ParameterSet.Config as cms

#TrackRefitter With Material
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitter.TrajectoryInEvent = True
# usually without refit: # TransientTrackingRecHitBuilder: no refit of hits...
#TrackRefitter.TTRHBuilder = 'WithoutRefit'
#from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
# ... but matching for strip stereo should be redone: 
#ttrhbwor.Matcher = 'StandardMatcher'

from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResidualsTier0 = MonitorTrackResiduals.clone(
    OutputMEsInRootFile = False,
    Mod_On = False
)

MonitorTrackResidualsDQM = MonitorTrackResiduals.clone(
    OutputMEsInRootFile = False,
    Mod_On = True
)

MonitorTrackResidualsStandAlone = MonitorTrackResiduals.clone(
    OutputMEsInRootFile = True,
    Mod_On = False
)

# Sequences
DQMMonitorTrackResidualsTier0 = cms.Sequence(TrackRefitter*MonitorTrackResidualsTier0)
DQMMonitorTrackResiduals = cms.Sequence(TrackRefitter*MonitorTrackResidualsDQM)
DQMMonitorTrackResidualsStandAlone = cms.Sequence(TrackRefitter*MonitorTrackResidualsStandAlone)


