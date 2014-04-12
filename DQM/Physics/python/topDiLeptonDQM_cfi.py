import FWCore.ParameterSet.Config as cms

topDiLeptonDQM = cms.EDAnalyzer("TopDiLeptonDQM",

    moduleName = cms.untracked.string('Physics/Top/DiLepton'),
    fileOutput = cms.bool(False),
    outputFile = cms.untracked.string('DiLeptonEvents.txt'),
    ### 
    TriggerResults = cms.InputTag('TriggerResults','','HLT'),
    hltPaths       = cms.vstring('HLT_Mu3','HLT_Mu5','HLT_Mu9','HLT_Mu15','HLT_IsoMu3','HLT_IsoMu9','HLT_DoubleMu0','HLT_DoubleMu3',
                                 'HLT_Ele10_LW_L1R','HLT_Ele15_LW_L1R','HLT_Ele20_LW_L1R'),
    ### 
    hltPaths_sig   = cms.vstring('HLT_Mu9', 'HLT_Mu9', 'HLT_IsoMu3', 'HLT_DoubleMu3', 'HLT_DoubleMu3', 'HLT_DoubleMu3'),
    hltPaths_trig  = cms.vstring('HLT_Mu3', 'HLT_Mu5', 'HLT_Mu3',    'HLT_Mu3',       'HLT_IsoMu3',    'HLT_DoubleMu0'),
    ### 
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    vertex_X_cut     = cms.double(  1.0 ),
    vertex_Y_cut     = cms.double(  1.0 ),
    vertex_Z_cut     = cms.double( 20.0 ),
    ### 
    muonCollection = cms.InputTag('muons'),
    muon_pT_cut    = cms.double( 1.0 ),
    muon_eta_cut   = cms.double( 2.4 ),
    muon_iso_cut   = cms.double( 0.2 ),
    ### 
    elecCollection = cms.InputTag('gedGsfElectrons'),
    elec_pT_cut    = cms.double( 5.0 ),
    elec_eta_cut   = cms.double( 2.4 ),
    elec_iso_cut   = cms.double( 0.2 ),
    elec_emf_cut   = cms.double( 0.1 ),
    ### 
    MassWindow_up   = cms.double( 106. ),
    MassWindow_down = cms.double(  76. )

)

topDiLeptonAnalyzer = cms.Sequence(topDiLeptonDQM)
