import FWCore.ParameterSet.Config as cms

HLTTopVal = cms.EDAnalyzer("TopValidation",
    OutputMEsInRootFile = cms.bool(False),
    TriggerResultsCollection = cms.InputTag("TriggerResults","","HLT"), 
    hltPaths = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3','HLT_Ele15_SW_L1R',
       'HLT_Ele15_SW_LooseTrackIso_L1R','HLT_DoubleEle10_SW_L1R'),
    hltMuonPaths = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3'),
    hltEgPaths = cms.vstring('HLT_Ele15_SW_L1R','HLT_Ele15_SW_LooseTrackIso_L1R','HLT_DoubleEle10_SW_L1R'),
    OutputFileName = cms.string(''),
    DQMFolder = cms.untracked.string("HLT/Top")
 )
