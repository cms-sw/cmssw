import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonQualityTests_cff import *
#include "DQMOffline/Ecal/data/ecal_dqm_client-offline.cff"
#include "DQM/SiStripMonitorClient/data/SiStripSourceConfigTier0.cff"
DQMOffline_SecondStep = cms.Sequence(muonQualityTests)

