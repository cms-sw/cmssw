import FWCore.ParameterSet.Config as cms

#commented out in 74X
#from DQM.HLTEvF.FourVectorHLTOnline_cfi import *

#from DQM.HLTEvF.OccupancyPlotter_cfi import *

from DQM.HLTEvF.HLTWorkspace_cfi import *
hlt4vector = cms.Path(hltWorkspace)

#hlt4vector = cms.Path(onlineOccPlot * hltWorkspace)
#hlt4vector = cms.Path(onlineOccPlot)
