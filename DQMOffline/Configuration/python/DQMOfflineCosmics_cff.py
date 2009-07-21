import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonCosmicMonitors_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_cosmic_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQMOffline.EGamma.cosmicPhotonAnalyzer_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_cff import *
from DQM.CSCMonitorModule.test.csc_dqm_sourceclient_offline_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *

DQMOfflineCosmics = cms.Sequence(SiStripDQMTier0*ecal_dqm_source_offline*muonCosmicMonitors*jetMETDQMOfflineSourceCosmic*hcalOfflineDQMSource*triggerOfflineDQMSource*siPixelOfflineDQM_cosmics_source*egammaCosmicPhotonMonitors*dtSources*cscSources*rpcTier0Source)

