import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TDTTPGClient_cfi import *
l1tdttpgseqClient = cms.Sequence(l1tdttpgClient)

