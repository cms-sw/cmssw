import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.HLTScouting.HLTScoutingDqmOffline_cff import *
from HLTriggerOffline.Scouting.ScoutingMuonMonitoring_Client_cff import *

hltScoutingMuonPostProcessing = cms.Sequence(muonEfficiencyNoVtx
                                             * muonEfficiencyVtx  
                                             * muonTriggerEfficiency
                                             )

hltScoutingPostProcessing = cms.Sequence(hltScoutingMuonPostProcessing)
