import FWCore.ParameterSet.Config as cms

# duplicated from HLTObjectMonitor_cff.py
# adding hltObjectMonitorPPReference for 2017 5 TeV PP reference run

from DQM.HLTEvF.HLTObjectMonitor_cfi import *
from DQM.HLTEvF.HLTObjectsMonitor_cfi import *
from DQM.HLTEvF.HLTLumiMonitoring_cff import *
from DQM.HLTEvF.HLTSiPixelMonitoring_cff import *
# strip monitoring@HLT needs track re-fitting (estimation of the crossing angle through the sensor)
# => some ESProducer modules have to be run
from DQM.HLTEvF.HLTSiStripMonitoring_cff import *
from DQM.HLTEvF.HLTTrackingMonitoring_cff import *
from DQM.HLTEvF.HLTPrimaryVertexMonitoring_cff import *
from DQM.HLTEvF.HLTHCALMonitoring_cff import *

from DQM.HLTEvF.HLTObjectMonitorPPReference_cfi import *
from DQM.HLTEvF.HLTFullIterTrackingMonitoring_cff import *


hlt4vector = cms.Path(
    lumiOnlineMonitorHLTsequence # lumi
    * hltObjectMonitor
    * hcalOnlineMonitoringSequence # HCAL monitoring
    * pixelOnlineMonitorHLTsequence # pixel cluster monitoring
    * sistripOnlineMonitorHLTsequence # strip cluster monitoring
    * trackingMonitoringHLTsequence # tracking monitoring
    * egmTrackingMonitorHLTsequence # EGM tracking monitoring
    * vertexingMonitorHLTsequence # vertexing
    * hltObjectsMonitor
    * hltObjectMonitorPPReference # hlt objects for ppRef run 2017
    * fullIterTracksMonitoringHLT # fullIterationTracks for ppRef run 2017
)
