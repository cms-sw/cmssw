import FWCore.ParameterSet.Config as cms

# Required services
InputTagDistributorService = cms.Service("InputTagDistributorService")
VariableHelperService = cms.Service("VariableHelperService")
UpdaterService = cms.Service("UpdaterService")
TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('ntuple.root'),
    )

# Common variables to ntuplizer
kine = cms.PSet(
    energy = cms.string('energy'),
    mass   = cms.string('mass'),
    mt     = cms.string('mt'),
    et     = cms.string('et'),
    pt     = cms.string('pt'),
    p      = cms.string('p'),
    px     = cms.string('px'),
    py     = cms.string('py'),
    pz     = cms.string('pz'),
    eta    = cms.string('eta'),
    phi    = cms.string('phi'),
    theta  = cms.string('theta'),
    )

# Collections to ntuplize
ntuplizeJetCollections = cms.EDFilter(
    "NTuplingDevice",
    Ntupler = cms.PSet(
    ComponentName = cms.string('StringBasedNTupler'),
    useTFileService = cms.bool(True), 
    branchesPSet = cms.PSet(
    treeName = cms.string('event'),

    # ---------- IC5 reco::GenJet collection ----------

    RecoGenJet = cms.PSet(
    Class = cms.string('reco::GenJet'),
    src = cms.InputTag('sortByGenJetPt:selectedLayer1Jets'),
    leaves = cms.PSet(kine),
    ),

    # ---------- IC5 reco::CaloJet collections ----------

    # Uncorrected
    RecoCaloJetIC5 = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:iterativeCone5CaloJets'),
    leaves = cms.PSet(kine),
    ),

    # ZSP-corrected
    RecoCaloJetZSP = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:ZSPJetCorJetIcone5'),
    leaves = cms.PSet(kine),
    ),

    # Default corrections
    RecoCaloJetJPTDefault = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloDefault'),
    leaves = cms.PSet(kine),
    ),

    # No corrections
    RecoCaloJetJPTNone = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloNone'),
    leaves = cms.PSet(kine),
    ),

    # In-cone only
    RecoCaloJetJPTInCone = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloInCone'),
    leaves = cms.PSet(kine),
    ),

    # + out-of-cone
    RecoCaloJetJPTOutOfCone = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloOutOfCone'),
    leaves = cms.PSet(kine),
    ),

    # + out-of-vertex
    RecoCaloJetJPTOutOfVertex = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloOutOfVertex'),
    leaves = cms.PSet(kine),
    ),

    # + pion efficiency
    RecoCaloJetJPTPionEff = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloPionEff'),
    leaves = cms.PSet(kine),
    ),

    # + muons
    RecoCaloJetJPTMuons = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloMuons'),
    leaves = cms.PSet(kine),
    ),

    # + electrons
    RecoCaloJetJPTElectrons = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloElectrons'),
    leaves = cms.PSet(kine),
    ),

    # + vectorial using tracks only
    RecoCaloJetJPTVecTracks = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloVecTracks'),
    leaves = cms.PSet(kine),
    ),
    
    # + vectorial using tracks and response
    RecoCaloJetJPTVecResponse = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:JPTCorJetIC5CaloVecResponse'),
    leaves = cms.PSet(kine),
    ),


    # ---------- IC5 pat::Jet collections ----------

    # Uncorrected
    PatJetIC5 = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:uncorrectedLayer1JetsIC5'),
    leaves = cms.PSet(kine),
    ),

    # ZSP-corrected
    PatJetZSP = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:PatZSPCorJetIC5Calo'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected
    PatJetJPT = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:PatJPTCorJetIC5Calo'),
    leaves = cms.PSet(kine),
    ),
    
    # JTA on-the-fly
    #PatJetJTA = cms.PSet(
    #Class = cms.string('reco::BasicJet'),
    #src = cms.InputTag('sortByGenJetPt:JTAJPTCorrectionIC5Calo'),
    #leaves = cms.PSet(kine),
    #),
    
    # MC-corrected
    PatJetMC = cms.PSet(
    Class = cms.string('reco::BasicJet'),
    src = cms.InputTag('sortByGenJetPt:selectedLayer1Jets'),
    leaves = cms.PSet(kine),
    ),

    ),
    ),
    )

