from DQM.TrackingMonitorClient.DQMScaleToClient_cfi import *

v0MonitorClient = dqmScaleToClient.clone()
v0MonitorClient.outputme = cms.PSet(
    folder = cms.string('Tracking/V0Monitoring'),
    name = cms.string('v0_Lxy_normalized'),
    factor = cms.double(1.)
)
v0MonitorClient.inputme = cms.PSet(
    folder = cms.string('Tracking/V0Monitoring'),
    name = cms.string('v0_Lxy')
)
