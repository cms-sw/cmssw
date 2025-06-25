import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *


jetDQMOnlineAnalyzerAk4ScoutingCleaned = jetDQMAnalyzerAk4ScoutingCleaned.clone()
jetDQMOnlineAnalyzerAk4ScoutingUncleaned = jetDQMAnalyzerAk4ScoutingUncleaned.clone()

jetDQMOnlineAnalyzerSequenceScouting = cms.Sequence(jetDQMOnlineAnalyzerAk4ScoutingUncleaned*jetDQMOnlineAnalyzerAk4ScoutingCleaned)

ScoutingJetMonitoring = cms.Sequence(jetPreDQMSeqScouting*
                                      dqmAk4PFScoutingL1FastL2L3ResidualCorrectorChain*
                                      jetDQMOnlineAnalyzerSequenceScouting)