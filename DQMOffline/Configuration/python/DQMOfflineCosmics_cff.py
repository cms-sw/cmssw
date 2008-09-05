import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonCosmicMonitors_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_cosmic_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *
from DQMOffline.JetMET.jetMETAnalyzer_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQMOffline.EGamma.cosmicPhotonAnalyzer_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *

DQMOfflineCosmics = cms.Sequence(SiStripDQMTier0*ecal_dqm_source_offline*muonCosmicMonitors*jetMETAnalyzer*hcalOfflineDQMSource*triggerOfflineDQMSource*siPixelOfflineDQM_source*egammaCosmicPhotonMonitors)

triggerOfflineDQMSource_woEmul = cms.Sequence(l1tmonitor*hltResults*egammaHLTDQM*HLTTauDQMOffline)

DQMOfflineCosmics_woEmul = cms.Sequence(SiStripDQMTier0*ecal_dqm_source_offline*muonCosmicMonitors*jetMETAnalyzer*hcalOfflineDQMSource*triggerOfflineDQMSource_woEmul*siPixelOfflineDQM_source*egammaCosmicPhotonMonitors)

