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

### RecHits monitoring
from HLTriggerOffline.Scouting.ScoutingRecHitAnalyzers_cff import *

hltScoutingMuonDqmOffline = cms.Sequence(scoutingMonitoringTagProbeMuonNoVtx *
                                         scoutingMonitoringTagProbeMuonVtx *
                                         scoutingMonitoringTriggerMuon_DoubleMu *
                                         scoutingMonitoringTriggerMuon_SingleMu *
                                         ScoutingMuonPropertiesMonitor )

hltScoutingJetDqmOffline = cms.Sequence(jetMETDQMOfflineSourceScouting)
## remove corrector to not schedule the run of the corrector modules which crash if scouting objects are missing
hltScoutingJetDqmOfflineForRelVals = cms.Sequence(jetMETDQMOfflineSourceScoutingNoCorrection)

hltScoutingCollectionMonitor = cms.Sequence(scoutingCollectionMonitor)

hltScoutingDqmOffline = cms.Sequence(hltScoutingMuonDqmOffline + hltScoutingEGammaDqmOffline + hltScoutingJetDqmOffline + hltScoutingCollectionMonitor + hltScoutingMonitoringRecHits)
hltScoutingDqmOfflineForRelVals = hltScoutingDqmOffline.copy()
hltScoutingDqmOfflineForRelVals.replace(hltScoutingJetDqmOffline, hltScoutingJetDqmOfflineForRelVals)
