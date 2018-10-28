import FWCore.ParameterSet.Config as cms

#commented out in 74X
#from DQM.HLTEvF.FourVectorHLTOnline_cfi import *
#from DQM.HLTEvF.OccupancyPlotter_cfi import *

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
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from DQM.HLTEvF.HLTObjectMonitorLeadLead_cfi import *
if pp_on_AA_2018:
    hlt4vector += hltObjectMonitorLeadLead

#hlt4vector = cms.Path(onlineOccPlot * hltObjectMonitor)
#hlt4vector = cms.Path(onlineOccPlot)
