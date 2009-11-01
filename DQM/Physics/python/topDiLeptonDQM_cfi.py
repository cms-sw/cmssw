import FWCore.ParameterSet.Config as cms

topDiLeptonDQM = cms.EDAnalyzer("TopDiLeptonDQM",

    moduleName     = cms.untracked.string('Physics/Top/DiLepton'),
    TriggerResults = cms.InputTag('TriggerResults','','HLT'),
    hltPaths_L3    = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3','HLT_Ele15_SW_L1R','HLT_Ele20_SW_L1R'),
    hltPaths_L3_mu = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3'),
    hltPaths_L3_el = cms.vstring('HLT_Ele15_SW_L1R','HLT_Ele20_SW_L1R'),

    muonCollection = cms.InputTag('muons'),
    muon_pT_cut    = cms.double(  4.0 ),
    muon_eta_cut   = cms.double(  5.0 ),

    elecCollection = cms.InputTag('gsfElectrons'),
    elec_pT_cut    = cms.double(  4.0 ),
    elec_eta_cut   = cms.double(  5.0 )

)

topDiLeptonAnalyzer = cms.Sequence(topDiLeptonDQM)
