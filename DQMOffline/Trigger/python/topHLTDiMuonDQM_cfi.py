import FWCore.ParameterSet.Config as cms

topHLTDiMuonDQM = cms.EDAnalyzer("TopHLTDiMuonDQM",

    monitorName    = cms.string('HLT/Top/HLTDiMuon/'),
    ### 
    TriggerResults = cms.InputTag('TriggerResults',       '','HLT'),
    TriggerEvent   = cms.InputTag('hltTriggerSummaryAOD', '','HLT'),
    ###  Trigger path: HLT_Mu15_v1
    TriggerFilter  = cms.InputTag('hltSingleMu15L3Filtered15','','HLT'),
    ###  Trigger path: HLT_Mu5
    # TriggerFilter  = cms.InputTag('hltSingleMu5L3Filtered5','','HLT'),
    ###  Trigger path: HLT_IsoMu3
    # TriggerFilter  = cms.InputTag('hltSingleMuIsoL3IsoFiltered3','','HLT'),
    ### 
    hltPaths_L1    = cms.vstring('HLT_L1MuOpen','HLT_L1Mu','HLT_L1Mu20','HLT_L1DoubleMuOpen'),
    hltPaths_L3    = cms.vstring('HLT_Mu9','HLT_Mu11','HLT_Mu13_v1','HLT_Mu15_v1',
                                 'HLT_IsoMu3','HLT_IsoMu9','HLT_DoubleMu3_v2','HLT_DoubleMu5_v1',
                                 'HLT_Ele10_LW_L1R','HLT_Ele15_LW_L1R','HLT_Ele20_LW_L1R'),
    ### 
    hltPaths       = cms.vstring('HLT_Mu9','HLT_Mu11','HLT_Mu13_v1','HLT_Mu15_v1',
                                 'HLT_IsoMu3','HLT_IsoMu9','HLT_DoubleMu3_v2','HLT_DoubleMu5_v1',
                                 'HLT_Ele10_LW_L1R','HLT_Ele15_LW_L1R','HLT_Ele20_LW_L1R'),
    hltPaths_sig   = cms.vstring('HLT_Mu9', 'HLT_Mu11', 'HLT_Mu15_v1', 'HLT_IsoMu9', 'HLT_DoubleMu3_v2', 'HLT_DoubleMu5_v1'),
    hltPaths_trig  = cms.vstring('HLT_Mu5', 'HLT_Mu5',  'HLT_Mu9',     'HLT_Mu9',    'HLT_Mu5',          'HLT_Mu5'),
    ### 
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    vertex_X_cut     = cms.double(  1.2 ),
    vertex_Y_cut     = cms.double(  1.2 ),
    vertex_Z_cut     = cms.double( 24.0 ),
    ### 
    muonCollection = cms.InputTag('muons'),
    muon_pT_cut    = cms.double( 5.0  ),
    muon_eta_cut   = cms.double( 2.5  ),
    muon_iso_cut   = cms.double( 0.15 ),
    ### 
    MassWindow_up   = cms.double( 106. ),
    MassWindow_down = cms.double(  76. )

)

topHLTDiMuonAnalyzer = cms.Sequence(topHLTDiMuonDQM)
