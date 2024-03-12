import FWCore.ParameterSet.Config as cms

#sometimes the dqm service is not already setup...
DQMStore = cms.Service("DQMStore")

from DQMServices.Components.MEtoEDMConverter_cfi import *
endOfProcess=cms.Sequence(MEtoEDMConverter)
# foo bar baz
# YjEJskazKk4TE
# 9OkveAEYb2MMy
