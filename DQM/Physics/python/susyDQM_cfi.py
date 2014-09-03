import FWCore.ParameterSet.Config as cms

susyDQM = cms.EDAnalyzer("RecoSusyDQM",

    moduleName     = cms.untracked.string('Physics/Susy'),

    muonCollection = cms.InputTag('muons'),
    electronCollection = cms.InputTag('gedGsfElectrons'),
    jetCollection = cms.InputTag('ak4CaloJets'),
    metCollection = cms.InputTag('caloMet'),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),

    muon_eta_cut   = cms.double(  2.4 ),
    muon_nHits_cut = cms.double(  11 ),
    muon_nChi2_cut = cms.double(  10 ),
    muon_d0_cut    = cms.double(  0.2 ),

    elec_eta_cut   = cms.double(  2.5 ),
    elec_mva_cut   = cms.double(  0.1 ),
    elec_d0_cut    = cms.double(  0.2 ),

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
