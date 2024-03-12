import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cff import *

muonFullOfflineDQM = cms.Sequence(
    hltMuonOfflineAnalyzers
)

muoHLTDQMSourceExtra = cms.Sequence(
)
# foo bar baz
# q7gcjgardyA4k
