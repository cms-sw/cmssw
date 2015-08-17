import FWCore.ParameterSet.Config as cms

#commented out in 74X
#from DQM.HLTEvF.FourVectorHLTOnline_cfi import *
#from DQM.HLTEvF.OccupancyPlotter_cfi import *

from DQM.HLTEvF.HLTObjectMonitor_cfi import *
# strip monitoring@HLT needs track re-fitting (estimation of the crossing angle through the sensor)
# => some ESProducer modules have to be run
# one of those (hltESPStripCPEfromTrackAngle) depends on the strip APV gain
# and therefore it has different setting in 25ns and 50ns HLT menu
# current setting is coherent w/ 50ns menu
# DQM.HLTEvF.HLTSiStripMonitoring_cff has to be updated as soon as the 25ns menu will be in production
from DQM.HLTEvF.HLTSiStripMonitoring_cff import *
from DQM.HLTEvF.HLTTrackingMonitoring_cff import *


hlt4vector = cms.Path(
    hltObjectMonitor
#    * sistripOnlineMonitorHLTsequence # strip cluster monitoring
    * trackingMonitoringHLTsequence # tracking monitoring
)


#hlt4vector = cms.Path(onlineOccPlot * hltObjectMonitor)
#hlt4vector = cms.Path(onlineOccPlot)
