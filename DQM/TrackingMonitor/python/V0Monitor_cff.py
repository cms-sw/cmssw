import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.V0Monitor_cfi import *

KshortMonitoring = v0Monitor.clone(
    FolderName = "Tracking/V0Monitoring/Ks",
    v0 = 'generalV0Candidates:Kshort',
    histoPSet = v0Monitor.histoPSet.clone(
        massPSet = v0Monitor.histoPSet.massPSet.clone(
            nbins = 100,
            xmin = 0.400,
            xmax = 0.600
        )
    )
)

LambdaMonitoring = v0Monitor.clone(
    FolderName = "Tracking/V0Monitoring/Lambda",
    v0 = 'generalV0Candidates:Lambda',
    histoPSet = v0Monitor.histoPSet.clone(
        massPSet = v0Monitor.histoPSet.massPSet.clone(
            nbins = 100,
            xmin = 1.050,
            xmax = 1.250
        )
    )
)

voMonitoringSequence = cms.Sequence(
    KshortMonitoring
    + LambdaMonitoring
)

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# tracker ON
KshortMonitoringCommon = KshortMonitoring.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
)
KshortMonitoringCommon.setLabel("KshortMonitoringCommon")

LambdaMonitoringCommon = LambdaMonitoring.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
)
LambdaMonitoringCommon.setLabel("LambdaMonitoringCommon")

voMonitoringCommonSequence = cms.Sequence(
    KshortMonitoringCommon
    + LambdaMonitoringCommon
)


# tracker ON + ZeroBias selection
KshortMonitoringMB = KshortMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_INpu/Ks",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
)
KshortMonitoringMB.setLabel("KshortMonitoringMB")

LambdaMonitoringMB = LambdaMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_INpu/Lambda",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
)
LambdaMonitoringMB.setLabel("LambdaMonitoringMB")

# tracker ON + no HIP no OOT selection (HLT_ZeroBias_FirstCollisionAfterAbortGap)
KshortMonitoringZBnoHIPnoOOT = KshortMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Ks",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb
)
KshortMonitoringZBnoHIPnoOOT.setLabel("KshortMonitoringZBnoHIPnoOOT")

LambdaMonitoringZBnoHIPnoOOT = LambdaMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Lambda",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb
)
LambdaMonitoringZBnoHIPnoOOT.setLabel("LambdaMonitoringZBnoHIPnoOOT")

# tracker ON + HIP no OOT selection (HLT_ZeroBias_FirstCollisionInTrain)
KshortMonitoringZBHIPnoOOT = KshortMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_noOOTpu_INpu/Ks",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb
)
KshortMonitoringZBHIPnoOOT.setLabel("KshortMonitoringZBHIPnoOOT")

LambdaMonitoringZBHIPnoOOT = LambdaMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_noOOTpu_INpu/Lambda",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb
)
LambdaMonitoringZBHIPnoOOT.setLabel("LambdaMonitoringZBHIPnoOOT")

# tracker ON + HIP OOT selection (HLT_ZeroBias_FirstBXAfterTrain)
KshortMonitoringZBHIPOOT = KshortMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_noINpu/Ks",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb
)
KshortMonitoringZBHIPOOT.setLabel("KshortMonitoringZBHIPOOT")

LambdaMonitoringZBHIPOOT = LambdaMonitoring.clone(
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_noINpu/Lambda",
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb
)
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
KshortWlxy16 = vertexCompositeCandidateCollectionSelector.clone(
    v0 = 'generalV0Candidates:Kshort'
)

LambdaWlxy16 = vertexCompositeCandidateCollectionSelector.clone(
    v0 = 'generalV0Candidates:Lambda'
)

KshortWlxy16Monitoring = KshortMonitoring.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/Ks/Lxy16"
)

LambdaWlxy16Monitoring = LambdaMonitoring.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/Lambda/Lxy16"
)

voWcutMonitoringSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16Monitoring
    + LambdaWlxy16*LambdaWlxy16Monitoring
)

KshortWlxy16MonitoringCommon = KshortMonitoringCommon.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/Ks/Lxy16"
)

LambdaWlxy16MonitoringCommon = LambdaMonitoringCommon.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/Lambda/Lxy16"
)

voWcutMonitoringCommonSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringCommon
    +LambdaWlxy16*LambdaWlxy16MonitoringCommon
)

KshortWlxy16MonitoringMB = KshortMonitoringMB.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_INpu/Ks/Lxy16"
)

LambdaWlxy16MonitoringMB = LambdaMonitoringMB.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_INpu/Lambda/Lxy16"
)

voWcutMonitoringMBSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringMB
    +LambdaWlxy16*LambdaWlxy16MonitoringMB
)

KshortWlxy16MonitoringZBnoHIPnoOOT = KshortMonitoringZBnoHIPnoOOT.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Ks/Lxy16"
)

LambdaWlxy16MonitoringZBnoHIPnoOOT = LambdaMonitoringZBnoHIPnoOOT.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/noHIP_noOOTpu_INpu/Lambda/Lxy16"
)

voWcutMonitoringZBnoHIPnoOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBnoHIPnoOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBnoHIPnoOOT    
)

KshortWlxy16MonitoringZBHIPnoOOT = KshortMonitoringZBHIPnoOOT.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_noOOTpu_INpu/Ks/Lxy16"
)

LambdaWlxy16MonitoringZBHIPnoOOT = LambdaMonitoringZBHIPnoOOT.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_noOOTpu_INpu/Lambda/Lxy16"
)

voWcutMonitoringZBHIPnoOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBHIPnoOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBHIPnoOOT
)

KshortWlxy16MonitoringZBHIPOOT = KshortMonitoringZBHIPOOT.clone(
    v0 = 'KshortWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_noINpu/Ks/Lxy16"
)

LambdaWlxy16MonitoringZBHIPOOT = LambdaMonitoringZBHIPOOT.clone(
    v0 = 'LambdaWlxy16',
    FolderName = "Tracking/V0Monitoring/HIP_OOTpu_noINpu/Lambda/Lxy16"
)

voWcutMonitoringZBHIPOOTSequence = cms.Sequence(
    KshortWlxy16*KshortWlxy16MonitoringZBHIPOOT
    +LambdaWlxy16*LambdaWlxy16MonitoringZBHIPOOT
)
