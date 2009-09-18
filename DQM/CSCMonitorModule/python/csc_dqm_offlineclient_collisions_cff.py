import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_offlineclient_cfi import *

cscOfflineCollisionsClients = cms.Sequence(dqmCSCOfflineClient)
