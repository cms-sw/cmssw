from DQM.TrackingMonitorClient.DQMScaleToClient_cfi import *

v0MonitorClient = dqmScaleToClient.clone(
    outputme = dqmScaleToClient.outputme.clone(
        folder = 'Tracking/V0Monitoring',
        name = 'v0_Lxy_normalized',
        factor = 1.
    ),
    inputme = dqmScaleToClient.inputme.clone(
        folder = 'Tracking/V0Monitoring',
        name = 'v0_Lxy'
    )
)
