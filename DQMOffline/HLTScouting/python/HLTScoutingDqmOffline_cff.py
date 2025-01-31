# ------------------------------------------- #
# Scouting DQM sequence for offline DQM       #
#                                             #
# used by DQM GUI: DQMOffline/Configuration   #
# ------------------------------------------- #
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester                                                                                                            

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester 

from HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi import *
from HLTriggerOffline.Scouting.ScoutingMuonTagProbeAnalyzer_cfi import *
from HLTriggerOffline.Scouting.ScoutingMuonMonitoring_Client_cff import *

from HLTriggerOffline.Scouting.HLTScoutingEGammaDqmOffline_cff import *

hltScoutingMuonDqmOffline = cms.Sequence(scoutingMonitoringTagProbeMuonNoVtx
        * scoutingMonitoringTagProbeMuonVtx                                                                                                                                              
        * scoutingMonitoringTriggerMuon                                                                                                                                                  
) 

hltScoutingDqmOffline = cms.Sequence(hltScoutingMuonDqmOffline + hltScoutingEGammaDqmOffline)
