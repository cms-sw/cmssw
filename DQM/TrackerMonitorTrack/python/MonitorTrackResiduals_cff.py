import FWCore.ParameterSet.Config as cms

#
# TrackRefitter 
#
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
import copy
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResidualsTier0 = copy.deepcopy(MonitorTrackResiduals)
import copy
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResidualsDQM = copy.deepcopy(MonitorTrackResiduals)
import copy
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResidualsStandAlone = copy.deepcopy(MonitorTrackResiduals)
DQMMonitorTrackResidualsTier0 = cms.Sequence(TrackRefitter*MonitorTrackResidualsTier0)
DQMMonitorTrackResiduals = cms.Sequence(TrackRefitter*MonitorTrackResidualsDQM)
DQMMonitorTrackResidualsStandAlone = cms.Sequence(TrackRefitter*MonitorTrackResidualsStandAlone)
TrackRefitter.src = 'ctfWithMaterialTracks'
TrackRefitter.TrajectoryInEvent = True
# usually without refit: 
TrackRefitter.TTRHBuilder = 'WithoutRefit'
# ... but matching for strip stereo should be redone: 
ttrhbwor.Matcher = 'StandardMatcher'
MonitorTrackResidualsTier0.OutputMEsInRootFile = False
MonitorTrackResidualsTier0.Mod_On = False
MonitorTrackResidualsDQM.OutputMEsInRootFile = False
MonitorTrackResidualsDQM.Mod_On = True
MonitorTrackResidualsStandAlone.OutputMEsInRootFile = True
MonitorTrackResidualsStandAlone.Mod_On = False

