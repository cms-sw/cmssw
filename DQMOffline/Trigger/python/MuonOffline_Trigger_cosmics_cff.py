import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi import *
from DQMOffline.Trigger.TopTrigRateAnalyzer_cosmics_cfi import *
from DQMOffline.Trigger.TnPEfficiency_cff import *


muonFullOfflineDQM = cms.Sequence( offlineDQMMuonTrigCosmics
								   + topTrigOfflineDQMCosmics
								   + TnPEfficiency )

