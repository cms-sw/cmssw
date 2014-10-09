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
    electronCollection       = cms.InputTag("gedGsfElectrons"),
    pfelectronCollectionEI   = cms.InputTag("pfIsolatedElectronsEI"),

    muonCollection           = cms.InputTag("muons"),
    pfmuonCollectionEI       = cms.InputTag("pfIsolatedMuonsEI"),

    tauCollection            = cms.InputTag("caloRecoTauProducer"),
    #pftauCollection          = cms.InputTag("pfTaus"),

    photonCollection         = cms.InputTag("gedPhotons"),
    #pfphotonCollection       = cms.InputTag("pfIsolatedPhotons"),

    caloJetCollection        = cms.InputTag("ak4CaloJetsCHS"),
    pfJetCollection          = cms.InputTag("ak4PFJetsCHS"),
    pfJetCollectionEI        = cms.InputTag("pfJets"),

    caloMETCollection        = cms.InputTag("caloMetM","","RECO"),
    pfMETCollection          = cms.InputTag("pfMet","","RECO"),
    pfMETCollectionEI        = cms.InputTag("pfMetEI","","RECO"),

    CaloJetCorService        = cms.string("ak4CaloL1FastL2L3"),
    PFJetCorService          = cms.string("ak4PFL1FastL2L3"),

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
    # Monojet
    monojet_PFJet_pt_cut      = cms.double(80.0),
    monojet_PFJet_met_cut     = cms.double(100.0),
    # MonoMuon
    monomuon_Muon_pt_cut      = cms.double(50.0),
    monomuon_Muon_met_cut     = cms.double(100.0),
    # MonoElectron
    monoelectron_Electron_pt_cut  = cms.double(50.0),
    monoelectron_Electron_met_cut = cms.double(80.0),

    #
    #LongLived

    #genParticleCollection    = cms.InputTag("genParticles"),

    #PtThrMu1 = cms.untracked.double(3.0),
    #PtThrMu2 = cms.untracked.double(3.0)

    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )

)
