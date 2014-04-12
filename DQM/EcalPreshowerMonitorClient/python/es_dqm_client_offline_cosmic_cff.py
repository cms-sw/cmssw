import FWCore.ParameterSet.Config as cms

from DQM.EcalPreshowerMonitorClient.EcalPreshowerMonitorClient_cfi import *

es_dqm_client_offline = cms.Sequence(ecalPreshowerMonitorClient)
