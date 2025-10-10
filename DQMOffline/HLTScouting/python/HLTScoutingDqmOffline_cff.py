'''
Scouting DQM sequences for offline DQM developed for 2025 pp data-taking
and used by DQM GUI (DQMOffline/Configuration):
currently running EGM and MUO monitoring modules.
'''

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester                                

### Muons monitoring
from HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi import *
from HLTriggerOffline.Scouting.ScoutingMuonTagProbeAnalyzer_cfi import *
from HLTriggerOffline.Scouting.ScoutingMuonPropertiesMonitoring_cfi import *

### Egamma monitoring
from HLTriggerOffline.Scouting.HLTScoutingEGammaDqmOffline_cff import *

### Jets Monitoring
from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *

### Miscellaneous monitoring
from DQM.HLTEvF.ScoutingCollectionMonitor_cfi import *

hltScoutingMuonDqmOffline = cms.Sequence(scoutingMonitoringTagProbeMuonNoVtx *
                                         scoutingMonitoringTagProbeMuonVtx *
                                         scoutingMonitoringTriggerMuon_DoubleMu *
                                         scoutingMonitoringTriggerMuon_SingleMu *
                                         ScoutingMuonPropertiesMonitor )

hltScoutingJetDqmOffline = cms.Sequence(jetMETDQMOfflineSourceScouting)

hltScoutingCollectionMonitor = cms.Sequence(scoutingCollectionMonitor)

hltScoutingDqmOffline = cms.Sequence(hltScoutingMuonDqmOffline + hltScoutingEGammaDqmOffline + hltScoutingJetDqmOffline +  hltScoutingCollectionMonitor)