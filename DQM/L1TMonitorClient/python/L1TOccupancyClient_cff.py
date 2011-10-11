import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TOccupancyClient_cfi import *

l1tOccupancyClientPath = cms.Sequence(l1tOccupancyClient)
