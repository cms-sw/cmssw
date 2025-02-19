import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.FourVectorHLTOnline_cfi import *

from DQM.HLTEvF.OccupancyPlotter_cfi import *

#hlt4vector = cms.Path(hltResultsOn * onlineOccPlot)
hlt4vector = cms.Path(onlineOccPlot)
