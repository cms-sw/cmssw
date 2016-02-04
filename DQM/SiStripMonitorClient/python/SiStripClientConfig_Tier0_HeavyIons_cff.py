import FWCore.ParameterSet.Config as cms

# import p+p sequence
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *

# clone and modify modules
siStripQTesterHI = siStripQTester.clone(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0_heavyions.xml')
    )

# define new HI sequence
SiStripOfflineDQMClientHI = cms.Sequence(siStripQTesterHI*siStripOfflineAnalyser)
