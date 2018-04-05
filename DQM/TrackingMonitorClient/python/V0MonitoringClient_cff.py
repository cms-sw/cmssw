import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.V0MonitoringClient_cfi import *

KshortMonitorClient = v0MonitorClient.clone()
KshortMonitorClient.inputme.folder  = cms.string('Tracking/V0Monitoring/Ks')
KshortMonitorClient.outputme.folder = cms.string('Tracking/V0Monitoring/Ks')

LambdaMonitoringClient = v0MonitorClient.clone()
LambdaMonitoringClient.inputme.folder  = cms.string('Tracking/V0Monitoring/Lambda')
LambdaMonitoringClient.outputme.folder = cms.string('Tracking/V0Monitoring/Lambda')

voMonitoringClientSequence = cms.Sequence(
    KshortMonitorClient
    + LambdaMonitoringClient
)

