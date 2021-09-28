import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.V0MonitoringClient_cfi import *

KshortMonitorClient = v0MonitorClient.clone(
    inputme = v0MonitorClient.inputme.clone(
        folder = 'Tracking/V0Monitoring/Ks'
    ),
    outputme = v0MonitorClient.outputme.clone(
        folder = 'Tracking/V0Monitoring/Ks'
    )
)

LambdaMonitoringClient = v0MonitorClient.clone(
    inputme = v0MonitorClient.inputme.clone(
        folder = 'Tracking/V0Monitoring/Lambda'
    ),
    outputme = v0MonitorClient.outputme.clone(
        folder = 'Tracking/V0Monitoring/Lambda'
    )
)

voMonitoringClientSequence = cms.Sequence(
    KshortMonitorClient
    + LambdaMonitoringClient
)

