import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cfi import *

from DQMOffline.MuonDPG.cscTnPEfficiencyTask_cfi import *

from DQMOffline.Muon.CSCMonitor_cfi import cscMonitor

cscSources = cms.Sequence(dqmCSCClient + 
                          cscTnPEfficiencyMonitor+
                          cscMonitor)
