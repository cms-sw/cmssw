'''
Scouting DQM sequences for offline DQM developed for 2025 pp data-taking
and used by DQM GUI (DQMOffline/Configuration):
currently running EGM and MUO monitoring modules.
'''

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.HLTScouting.HLTScoutingDqmOffline_cff import *

from HLTriggerOffline.Scouting.ScoutingMuonMonitoring_Client_cff import *
from HLTriggerOffline.Scouting.HLTScoutingEGammaPostProcessing_cff import *

hltScoutingMuonPostProcessing = cms.Sequence(muonEfficiencyNoVtx
                                             * muonEfficiencyVtx  
                                             * muonTriggerEfficiency
                                             )

hLTScoutingPostProcessing = cms.Sequence(hltScoutingMuonPostProcessing + hltScoutingEGammaPostProcessing)
