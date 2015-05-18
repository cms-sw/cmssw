import FWCore.ParameterSet.Config as cms

#commented out in 74X
#from DQM.HLTEvF.FourVectorHLTOnline_cfi import *
#from DQM.HLTEvF.OccupancyPlotter_cfi import *

from DQM.HLTEvF.HLTObjectMonitor_cfi import *

hlt4vector = cms.Path(
    hltObjectMonitor
)


#hlt4vector = cms.Path(onlineOccPlot * hltObjectMonitor)
#hlt4vector = cms.Path(onlineOccPlot)
