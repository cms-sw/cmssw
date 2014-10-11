import FWCore.ParameterSet.Config as cms

ExoticaDQM = cms.EDAnalyzer(
    "ExoticaDQM",

    #Trigger Results
    triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),

    #Trigger Lists
    triggerMultiJetsList     = cms.vstring(
    "HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v"
    ),
    triggerLongLivedList     = cms.vstring(
    "HLT_L2DoubleMu23_NoVertex_v"
    "HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v"
    "HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v"
    "HLT_HT650_Track50_dEdx3p6_v"
    "HLT_HT650_Track60_dEdx3p7_v"
    "HLT_MET80_Track50_dEdx3p6_v"
    "HLT_MET80_Track60_dEdx3p7_v"
    "HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v"
    "HLT_HT300_SingleDisplacedPFJet60_v"
    "HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v"
    "HLT_HT300_DoubleDisplacedPFJet60_v"
    "HLT_JetE30_NoBPTX_v"
    "HLT_JetE30_NoBPTX3BX_NoHalo_v"
    "HLT_JetE50_NoBPTX3BX_NoHalo_v"
    "HLT_JetE70_NoBPTX3BX_NoHalo_v"
    "HLT_Mu40_eta2p1_Track50_dEdx3p6_v"
    "HLT_Mu40_eta2p1_Track60_dEdx3p7_v"
    "HLT_L2Mu70_eta2p1_PFMET55_v"
    "HLT_L2Mu70_eta2p1_PFMET60_v"
    "HLT_L2Mu20_eta2p1_NoVertex_v"
    "HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v"
    "HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v"
    "HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v"
    "HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v"
    "HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v"
    "HLT_DoublePhoton48_HEVT_v"
    "HLT_DoublePhoton53_HEVT_v"
    "HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v"
    "HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v"
    ),

    #Physics objects
    vertexCollection         = cms.InputTag('offlinePrimaryVertices'),

    electronCollection       = cms.InputTag("gedGsfElectrons"),

    muonCollection           = cms.InputTag("muons"),

    photonCollection         = cms.InputTag("gedPhotons"),

    pfJetCollection          = cms.InputTag('ak4PFJetsCHS'),
    PFJetCorService          = cms.string("ak4PFL1FastL2L3"),

    DiJetPFJetCollection     = cms.VInputTag('ak4PFJetsCHS','ak8PFJetsCHS','ca8PFJetsCHS'),

    caloMETCollection        = cms.InputTag("caloMetM","","RECO"),
    pfMETCollection          = cms.InputTag("pfMet","","RECO"),


    #Cuts
    # DiJet
    dijet_PFJet1_pt_cut       = cms.double(30.0),
    dijet_PFJet2_pt_cut       = cms.double(30.0),
    # DiMuon
    dimuon_Muon1_pt_cut      = cms.double(50.0),
    dimuon_Muon2_pt_cut      = cms.double(50.0),
    # DiElectron
    dielectron_Electron1_pt_cut = cms.double(50.0),
    dielectron_Electron2_pt_cut = cms.double(50.0),
    # DiPhoton
    diphoton_Photon1_pt_cut   = cms.double(20.0),
    diphoton_Photon2_pt_cut   = cms.double(20.0),
    # MonoMuon
    monomuon_Muon_pt_cut      = cms.double(80.0),
    monomuon_Muon_met_cut     = cms.double(100.0),
    # MonoElectron
    monoelectron_Electron_pt_cut  = cms.double(80.0),
    monoelectron_Electron_met_cut = cms.double(100.0),
    # Monojet
    monojet_PFJet_pt_cut      = cms.double(80.0),
    monojet_PFJet_met_cut     = cms.double(100.0),
    # MonoPhoton
    monophoton_Photon_pt_cut  = cms.double(80.0),
    monophoton_Photon_met_cut = cms.double(100.0),

    
    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )

)
