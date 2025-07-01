import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *

jetDQMOnlineAnalyzerAk4ScoutingCleaned = jetDQMAnalyzerAk4ScoutingCleaned.clone(
    JetType='scoutingOnline',
    DCSFilterForJetMonitoring=dict(DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
                                   onlineMetaDataDigisSrc = cms.untracked.InputTag("hltOnlineMetaDataDigis"),
                                   DebugOn = cms.untracked.bool(False),
                                   alwaysPass = False)
)

jetDQMOnlineAnalyzerAk4ScoutingUncleaned = jetDQMAnalyzerAk4ScoutingUncleaned.clone(
    JetType='scoutingOnline',
    DCSFilterForJetMonitoring=dict(DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
                                   onlineMetaDataDigisSrc = cms.untracked.InputTag("hltOnlineMetaDataDigis"),
                                   DebugOn =  cms.untracked.bool(False),
                                   alwaysPass = False)
)

jetDQMOnlineAnalyzerSequenceScouting = cms.Sequence(jetDQMOnlineAnalyzerAk4ScoutingUncleaned*
                                                    jetDQMOnlineAnalyzerAk4ScoutingCleaned)

ScoutingJetMonitoring = cms.Sequence(jetPreDQMSeqScouting*
                                     dqmAk4PFScoutingL1FastL2L3ResidualCorrectorChain*
                                     jetDQMOnlineAnalyzerSequenceScouting)
