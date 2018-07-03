import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TStage2CaloLayer2DEClient_cfi import *

l1t2016Clients = cms.Sequence(
    l1tStage2CaloLayer2DEClient
)
