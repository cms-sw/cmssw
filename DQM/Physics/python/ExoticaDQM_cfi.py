import FWCore.ParameterSet.Config as cms

ExoticaDQM = cms.EDAnalyzer(
    "ExoticaDQM",

    #Trigger Results
    TriggerResults           = cms.InputTag('TriggerResults','','HLT'),
#    HltPaths                 = cms.vstring('HLT_*'),
    HltPaths                 = cms.vstring("HLT_Mu","HLT_Ele","HLT_Photon","HLT_PFHT","HLT_HT","HLT_PFMET","HLT_MET","HLT_"),

    #Physics objects
    vertexCollection         = cms.InputTag('offlinePrimaryVertices'),
    electronCollection       = cms.InputTag("gedGsfElectrons"),

    muonCollection           = cms.InputTag("muons"),

    photonCollection         = cms.InputTag("gedPhotons"),

    pfJetCollection          = cms.InputTag('ak4PFJetsCHS'),
    corrector                = cms.InputTag('ak4PFL1FastL2L3Corrector'),

    DiJetPFJetCollection     = cms.VInputTag('ak4PFJetsCHS','ak8PFJetsCHS'),

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
