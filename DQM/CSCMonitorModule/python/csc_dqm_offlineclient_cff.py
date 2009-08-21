import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_offlineclient_cfi import *

cscOfflineSources = cms.Sequence(dqmCSCOfflineClient)
