import FWCore.ParameterSet.Config as cms

topDiLeptonDQM = cms.EDAnalyzer("TopDiLeptonDQM",

    moduleName     = cms.untracked.string('Physics/Top/DiLepton'),
    ### 
    TriggerResults = cms.InputTag('TriggerResults','','HLT'),
    hltPaths       = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3','HLT_Ele15_SW_L1R','HLT_Ele20_SW_L1R'),
    ### 
    hltPaths_mu    = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3'),
    hltPaths_el    = cms.vstring('HLT_Ele15_SW_L1R','HLT_Ele20_SW_L1R'),
    ### 
    hltPaths_sig   = cms.vstring('HLT_IsoMu9', 'HLT_Mu15', 'HLT_DoubleMu3', 'HLT_Mu9'),
    hltPaths_trig  = cms.vstring('HLT_Mu9',    'HLT_Mu9',  'HLT_Mu9',       'HLT_Mu5'),
    ### 
    muonCollection = cms.InputTag('muons'),
    muon_pT_cut    = cms.double(  3.0 ),
    muon_eta_cut   = cms.double(  5.0 ),
    muon_iso_cut   = cms.double(  0.2 ),
    ### 
    elecCollection = cms.InputTag('gsfElectrons'),
    elec_pT_cut    = cms.double(  3.0 ),
    elec_eta_cut   = cms.double(  5.0 ),
    elec_iso_cut   = cms.double(  0.2 ),
    ### 
    MassWindow_up   = cms.double( 1000. ),
    MassWindow_down = cms.double(    0. )

)

topDiLeptonAnalyzer = cms.Sequence(topDiLeptonDQM)
