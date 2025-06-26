import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *

jetDQMOnlineAnalyzerAk4ScoutingCleaned = jetDQMAnalyzerAk4ScoutingCleaned.clone(
    DCSFilterForJetMonitoring = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        onlineMetaDataDigisSrc =  cms.untracked.InputTag("hltOnlineMetaDataDigis"),
        DebugOn = cms.untracked.bool(False),
        alwaysPass = cms.untracked.bool(False)
    )
)
jetDQMOnlineAnalyzerAk4ScoutingUncleaned = jetDQMAnalyzerAk4ScoutingUncleaned.clone(
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        onlineMetaDataDigisSrc =  cms.untracked.InputTag("hltOnlineMetaDataDigis"),
        DebugOn = cms.untracked.bool(False),
        alwaysPass = cms.untracked.bool(False)
    )
)

jetDQMOnlineAnalyzerSequenceScouting = cms.Sequence(jetDQMOnlineAnalyzerAk4ScoutingUncleaned*jetDQMOnlineAnalyzerAk4ScoutingCleaned)

ScoutingJetMonitoring = cms.Sequence(jetPreDQMSeqScouting*
                                     dqmAk4PFScoutingL1FastL2L3ResidualCorrectorChain*
                                     jetDQMOnlineAnalyzerSequenceScouting)
