import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cfi import *

from DQMOffline.MuonDPG.cscTnPEfficiencyTask_cfi import *

cscSources = cms.Sequence(dqmCSCClient + 
                          cscTnPEfficiencyMonitor)
