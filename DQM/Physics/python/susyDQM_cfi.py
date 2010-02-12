import FWCore.ParameterSet.Config as cms

susyDQM = cms.EDAnalyzer("RecoSusyDQM",

    moduleName     = cms.untracked.string('Physics/Susy'),

    muonCollection = cms.InputTag('muons'),
    electronCollection = cms.InputTag('gsfElectrons'),
    jetCollection = cms.InputTag('ak5CaloJets'),
    metCollection = cms.InputTag('met'),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),

    muon_eta_cut   = cms.double(  2.4 ),
    muon_nHits_cut = cms.double(  11 ),
    muon_nChi2_cut = cms.double(  10 ),
    muon_d0_cut    = cms.double(  0.2 ),

    elec_eta_cut   = cms.double(  2.5 ),
    elec_mva_cut   = cms.double(  0.1 ),
    elec_d0_cut    = cms.double(  0.2 ),

    RA12_muon_pt_cut    = cms.double(  10.0 ),
    RA12_muon_iso_cut   = cms.double(  0.1 ),

    RA12_elec_pt_cut    = cms.double( 15.0 ),
    RA12_elec_iso_cut   = cms.double(  0.5 ),
    
    RA1_jet_pt_cut      = cms.double(  30.0 ),
    RA1_jet_eta_cut     = cms.double(   5.0 ),
    RA1_jet_min_emf_cut = cms.double(   0.05 ),
    RA1_jet_max_emf_cut = cms.double(   0.95 ),
    RA1_jet1_pt_cut     = cms.double(  30.0 ),
    RA1_jet1_eta_cut    = cms.double(   2.5 ),
    RA1_jet2_pt_cut     = cms.double(  30.0 ),
    RA1_jet2_eta_cut    = cms.double(   2.5 ),
    RA1_jet3_pt_cut     = cms.double(  30.0 ),
    
    RA1_alphat_cut       = cms.double(     0.5 ),
    RA1_ht_cut           = cms.double(   200. ),
    RA1_mht_cut          = cms.double(     0. ),
    RA1_deltaPhi_cut     = cms.double(     0.0 ),
    RA1_deltaPhiJets_cut = cms.double(     3.0 ),

    RA2_jet_pt_cut      = cms.double(  30.0 ),
    RA2_jet_eta_cut     = cms.double(   5.0 ),
    RA2_jet_min_emf_cut = cms.double(   0.05 ),
    RA2_jet_max_emf_cut = cms.double(   0.95 ),
    RA2_jet1_pt_cut     = cms.double( 100.0 ),
    RA2_jet2_pt_cut     = cms.double(  50.0 ),
    RA2_jet3_pt_cut     = cms.double(  30.0 ),
    RA2_jet1_eta_cut     = cms.double(   1.2 ),
    RA2_jet2_eta_cut     = cms.double(   2.5 ),
    RA2_jet3_eta_cut     = cms.double(   5.0 ),
    RA2_N_jets_cut      = cms.int32(    3 ),
    
    RA2_ht_cut        = cms.double( 200.0 ),
    RA2_mht_cut       = cms.double( 100.0 ),
    RA2_deltaPhi_cut  = cms.double(   0.3 ),

    RAL_muon_pt_cut    = cms.double(  7.0 ),
    RAL_muon_iso_cut   = cms.double(  0.1 ),

    RAL_elec_pt_cut    = cms.double(  7.0 ),
    RAL_elec_iso_cut   = cms.double(  0.5 ),
    
    RAL_jet_pt_cut    = cms.double(  30.0 ),
    RAL_jet_eta_cut    = cms.double(  3.0 ),
    RAL_jet_min_emf_cut    = cms.double(  0.05 ),
    RAL_jet_max_emf_cut    = cms.double(  0.95 ),
    RAL_jet_sum_pt_cut    = cms.double(  100.0 ),
    
    RAL_met_cut    = cms.double(  50.0 )

)

susyAnalyzer = cms.Sequence(susyDQM)
