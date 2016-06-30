import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.V0Monitor_cfi import *

KshortMonitor = v0Monitor.clone()
KshortMonitor.FolderName = cms.string("Tracking/V0Monitoring/Ks")
KshortMonitor.v0         = cms.InputTag('generalV0Candidates:Kshort')
KshortMonitor.histoPSet.massPSet = cms.PSet(
   nbins = cms.int32 ( 100 ),
   xmin  = cms.double( 0.400),
   xmax  = cms.double( 0.600),
)

LambdaMonitoring = v0Monitor.clone()
LambdaMonitoring.FolderName = cms.string("Tracking/V0Monitoring/Lambda")
LambdaMonitoring.v0         = cms.InputTag('generalV0Candidates:Lambda')
LambdaMonitoring.histoPSet.massPSet = cms.PSet(
   nbins = cms.int32 ( 100 ),
   xmin  = cms.double( 1.050 ),
   xmax  = cms.double( 1.250 )
)

voMonitoringSequence = cms.Sequence(
    KshortMonitor
    + LambdaMonitoring
)


from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# tracker ON
KshortMonitorCommon = KshortMonitor.clone()
KshortMonitorCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
KshortMonitorCommon.setLabel("KshortMonitorCommon")

LambdaMonitoringCommon = LambdaMonitoring.clone()
LambdaMonitoringCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
LambdaMonitoringCommon.setLabel("LambdaMonitoringCommon")

voMonitoringCommonSequence = cms.Sequence(
    KshortMonitorCommon
    + LambdaMonitoringCommon
)


# tracker ON + ZeroBias selection
KshortMonitorMB = KshortMonitor.clone()
KshortMonitorMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
KshortMonitorMB.setLabel("KshortMonitorMB")

LambdaMonitoringMB = LambdaMonitoring.clone()
LambdaMonitoringMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
LambdaMonitoringMB.setLabel("LambdaMonitoringMB")

voMonitoringMBSequence = cms.Sequence(
    KshortMonitorMB
    + LambdaMonitoringMB
)

