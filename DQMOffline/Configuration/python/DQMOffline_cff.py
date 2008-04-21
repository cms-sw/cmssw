import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.Ecal.ecal_dqm_sourceclient_offline_cff import *
DQMOffline = cms.Sequence(ecal_dqm_sourceclient-offline1*ecal_dqm_sourceclient-offline2*muonMonitorsAndQualityTests)

