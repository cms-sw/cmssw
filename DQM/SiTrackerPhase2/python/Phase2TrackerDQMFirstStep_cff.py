import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cff import *
from DQM.SiTrackerPhase2.Phase2ITMonitorRecHit_cff import *
from DQM.SiTrackerPhase2.Phase2ITMonitorCluster_cff import *
from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cff import *
from DQM.SiTrackerPhase2.Phase2OTMonitorVectorHits_cff import *
#L1 
from DQM.SiTrackerPhase2.Phase2OTMonitorTTTrack_cfi import *
from DQM.SiTrackerPhase2.Phase2OTMonitorTTStub_cfi import *
from DQM.SiTrackerPhase2.Phase2OTMonitorTTCluster_cfi import *

trackerphase2DQMSource = cms.Sequence( pixDigiMon 
                                       + otDigiMon
                                       +rechitMonitorIT
                                       + clusterMonitorIT
                                       + clusterMonitorOT
                                       + Phase2OTMonitorTTCluster
                                       + Phase2OTMonitorTTStub
                                       + Phase2OTMonitorTTTrack
)

from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
trackerphase2DQMSource_wVhits = trackerphase2DQMSource.copy()
trackerphase2DQMSource_wVhits += cms.Sequence(acceptedVecHitsmon + rejectedVecHitsmon)

vectorHits.toReplaceWith(trackerphase2DQMSource, trackerphase2DQMSource_wVhits)
