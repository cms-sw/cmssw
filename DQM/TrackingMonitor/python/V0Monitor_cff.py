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



from CommonTools.RecoAlgos.vertexCompositeCandidateCollectionSelector_cfi import *
KshortWlxy16 = vertexCompositeCandidateCollectionSelector.clone()
KshortWlxy16.v0 = cms.InputTag('generalV0Candidates:Kshort')

LambdaWlxy16 = vertexCompositeCandidateCollectionSelector.clone()
LambdaWlxy16.v0 = cms.InputTag('generalV0Candidates:Lambda')

KshortWlxy16Monitoring = KshortMonitoring.clone()
KshortWlxy16Monitoring.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16Monitoring.FolderName = cms.string("Tracking/V0Monitoring/Ks/Lxy16")

LambdaWlxy16Monitoring = LambdaMonitoring.clone()
LambdaWlxy16Monitoring.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16Monitoring.FolderName = cms.string("Tracking/V0Monitoring/Lambda/Lxy16")

voWcutMonitoringSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16Monitoring
    + LambdaWlxy16*LambdaWlxy16Monitoring
)

KshortWlxy16MonitoringCommon = KshortMonitoringCommon.clone()
KshortWlxy16MonitoringCommon.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16MonitoringCommon.FolderName = cms.string("Tracking/V0Monitoring/Ks/Lxy16")

LambdaWlxy16MonitoringCommon = LambdaMonitoringCommon.clone()
LambdaWlxy16MonitoringCommon.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16MonitoringCommon.FolderName = cms.string("Tracking/V0Monitoring/Lambda/Lxy16")

voWcutMonitoringCommonSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringCommon
    +LambdaWlxy16*LambdaWlxy16MonitoringCommon
)

KshortWlxy16MonitoringMB = KshortMonitoringMB.clone()
KshortWlxy16MonitoringMB.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16MonitoringMB.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_INpu/Ks/Lxy16")

LambdaWlxy16MonitoringMB = LambdaMonitoringMB.clone()
LambdaWlxy16MonitoringMB.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16MonitoringMB.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_INpu/Lambda/Lxy16")

voWcutMonitoringMBSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringMB
    +LambdaWlxy16*LambdaWlxy16MonitoringMB
)

KshortWlxy16MonitoringZBnoHIPnoOOT = KshortMonitoringZBnoHIPnoOOT.clone()
KshortWlxy16MonitoringZBnoHIPnoOOT.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16MonitoringZBnoHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Ks/Lxy16")

LambdaWlxy16MonitoringZBnoHIPnoOOT = LambdaMonitoringZBnoHIPnoOOT.clone()
LambdaWlxy16MonitoringZBnoHIPnoOOT.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16MonitoringZBnoHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Lambda/Lxy16")


voWcutMonitoringZBnoHIPnoOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBnoHIPnoOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBnoHIPnoOOT    
)

KshortWlxy16MonitoringZBHIPnoOOT = KshortMonitoringZBHIPnoOOT.clone()
KshortWlxy16MonitoringZBHIPnoOOT.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16MonitoringZBHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_noOOTpu_INpu/Ks/Lxy16")

LambdaWlxy16MonitoringZBHIPnoOOT = LambdaMonitoringZBHIPnoOOT.clone()
LambdaWlxy16MonitoringZBHIPnoOOT.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16MonitoringZBHIPnoOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_noOOTpu_INpu/Lambda/Lxy16")

voWcutMonitoringZBHIPnoOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBHIPnoOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBHIPnoOOT
)

KshortWlxy16MonitoringZBHIPOOT = KshortMonitoringZBHIPOOT.clone()
KshortWlxy16MonitoringZBHIPOOT.v0 = cms.InputTag('KshortWlxy16')
KshortWlxy16MonitoringZBHIPOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_noINpu/Ks/Lxy16")

LambdaWlxy16MonitoringZBHIPOOT = LambdaMonitoringZBHIPOOT.clone()
LambdaWlxy16MonitoringZBHIPOOT.v0 = cms.InputTag('LambdaWlxy16')
LambdaWlxy16MonitoringZBHIPOOT.FolderName = cms.string("Tracking/V0Monitoring/HIP_OOTpu_noINpu/Lambda/Lxy16")

voWcutMonitoringZBHIPOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBHIPOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBHIPOOT
)
