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
from DQMOffline.Trigger.JetMETPromptMonitor_cff import *

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
hltScoutingJetDqmOfflineForRelVals = cms.Sequence(jetMETDQMOfflineSourceScoutingNoCorrection +
                                                  jetmetScoutingNoJECsMonitorHLT)

hltScoutingCollectionMonitor = cms.Sequence(scoutingCollectionMonitor)

hltScoutingDqmOffline = cms.Sequence(hltScoutingMuonDqmOffline +
                                     hltScoutingEGammaDqmOffline +
                                     hltScoutingJetDqmOffline +
                                     jetmetScoutingMonitorHLT +
                                     hltScoutingCollectionMonitor)

## Add the scouting rechits monitoring (only for 2025, integrated in menu GRun 2025 V1.3)
## See https://its.cern.ch/jira/browse/CMSHLT-3607
_hltScoutingDqmOffline = hltScoutingDqmOffline.copy()
_hltScoutingDqmOffline += hltScoutingMonitoringRecHits

# Append the RecHits monitoring only in the 2025 scouting era
from Configuration.Eras.Modifier_run3_scouting_2025_cff import run3_scouting_2025
run3_scouting_2025.toReplaceWith(hltScoutingDqmOffline, _hltScoutingDqmOffline)

hltScoutingDqmOfflineForRelVals = hltScoutingDqmOffline.copy()
hltScoutingDqmOfflineForRelVals.replace(hltScoutingJetDqmOffline, hltScoutingJetDqmOfflineForRelVals)
