import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.V0Monitor_cfi import *

KshortMonitoring = v0Monitor.clone()
KshortMonitoring.FolderName = cms.string("Tracking/V0Monitoring/Ks")
KshortMonitoring.v0         = cms.InputTag('generalV0Candidates:Kshort')
KshortMonitoring.histoPSet.massPSet = cms.PSet(
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
    KshortMonitoring
    + LambdaMonitoring
)


from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# tracker ON
KshortMonitoringCommon = KshortMonitoring.clone()
KshortMonitoringCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
KshortMonitoringCommon.setLabel("KshortMonitoringCommon")

LambdaMonitoringCommon = LambdaMonitoring.clone()
LambdaMonitoringCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
LambdaMonitoringCommon.setLabel("LambdaMonitoringCommon")

voMonitoringCommonSequence = cms.Sequence(
    KshortMonitoringCommon
    + LambdaMonitoringCommon
)


# tracker ON + ZeroBias selection
KshortMonitoringMB = KshortMonitoring.clone()
KshortMonitoringMB.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_INpu/Ks")
KshortMonitoringMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
KshortMonitoringMB.setLabel("KshortMonitoringMB")

LambdaMonitoringMB = LambdaMonitoring.clone()
LambdaMonitoringMB.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_INpu/Lambda")
LambdaMonitoringMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
LambdaMonitoringMB.setLabel("LambdaMonitoringMB")

# tracker ON + no HIP no OOT selection (HLT_ZeroBias_FirstCollisionAfterAbortGap)
KshortMonitoringZBnoHIPnoOOT = KshortMonitoring.clone()
KshortMonitoringZBnoHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Ks")
KshortMonitoringZBnoHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb
KshortMonitoringZBnoHIPnoOOT.setLabel("KshortMonitoringZBnoHIPnoOOT")

LambdaMonitoringZBnoHIPnoOOT = LambdaMonitoring.clone()
LambdaMonitoringZBnoHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Lambda")
LambdaMonitoringZBnoHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb
LambdaMonitoringZBnoHIPnoOOT.setLabel("LambdaMonitoringZBnoHIPnoOOT")

# tracker ON + HIP no OOT selection (HLT_ZeroBias_FirstCollisionInTrain)
KshortMonitoringZBHIPnoOOT = KshortMonitoring.clone()
KshortMonitoringZBHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_noOOTpu_INpu/Ks")
KshortMonitoringZBHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb
KshortMonitoringZBHIPnoOOT.setLabel("KshortMonitoringZBHIPnoOOT")

LambdaMonitoringZBHIPnoOOT = LambdaMonitoring.clone()
LambdaMonitoringZBHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_noOOTpu_INpu/Lambda")
LambdaMonitoringZBHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb
LambdaMonitoringZBHIPnoOOT.setLabel("LambdaMonitoringZBHIPnoOOT")

# tracker ON + HIP OOT selection (HLT_ZeroBias_FirstBXAfterTrain)
KshortMonitoringZBHIPOOT = KshortMonitoring.clone()
KshortMonitoringZBHIPOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_noINpu/Ks")
KshortMonitoringZBHIPOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb
KshortMonitoringZBHIPOOT.setLabel("KshortMonitoringZBHIPOOT")

LambdaMonitoringZBHIPOOT = LambdaMonitoring.clone()
LambdaMonitoringZBHIPOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_noINpu/Lambda")
LambdaMonitoringZBHIPOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb
LambdaMonitoringZBHIPOOT.setLabel("LambdaMonitoringZBHIPOOT")

voMonitoringMBSequence = cms.Sequence(
    KshortMonitoringMB
    + LambdaMonitoringMB
)

voMonitoringZBnoHIPnoOOTSequence = cms.Sequence(
    KshortMonitoringZBnoHIPnoOOT
    + LambdaMonitoringZBnoHIPnoOOT
)

voMonitoringZBHIPnoOOTSequence = cms.Sequence(
    KshortMonitoringZBHIPnoOOT
    + LambdaMonitoringZBHIPnoOOT
)

voMonitoringZBHIPOOTSequence = cms.Sequence(
    KshortMonitoringZBHIPOOT
    + LambdaMonitoringZBHIPOOT
)

