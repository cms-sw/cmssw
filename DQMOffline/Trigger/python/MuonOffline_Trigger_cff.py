import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cff import *
from DQMOffline.Trigger.topHLTDiMuonDQM_cfi import *

muonFullOfflineDQM = cms.Sequence(
    hltMuonOfflineAnalyzers
    + topHLTDiMuonAnalyzer
)
