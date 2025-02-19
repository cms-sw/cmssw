import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TEMUEventInfoClient_cfi import *
l1EmulatorEventInfoClient = cms.Sequence(l1temuEventInfoClient)


