import FWCore.ParameterSet.Config as cms

# Required services
InputTagDistributorService = cms.Service("InputTagDistributorService")
VariableHelperService = cms.Service("VariableHelperService")
UpdaterService = cms.Service("UpdaterService")

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
    useTFileService = cms.bool(False), 
    branchesPSet = cms.PSet(
    treeName = cms.string('event'),

    # reco::GenJets
    RecoGenJet = cms.PSet(
    Class = cms.string('reco::GenJet'),
    src = cms.InputTag('sort:cleanLayer1Jets'),
    leaves = cms.PSet(kine),
    ),

    # Uncorrected reco::CaloJets (IC5)
    RecoCaloJetIC5 = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:iterativeCone5CaloJets'),
    leaves = cms.PSet(kine),
    ),

    # ZSP-corrected reco::CaloJets
    RecoCaloJetZSP = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:ZSPJetCorJetIcone5'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (old JPT)
    RecoCaloJetJPT = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JetPlusTrackZSPCorJetIcone5'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (new JPT, default corrections)
    RecoCaloJetJPTDefault = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorDefault'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (no corrections)
    RecoCaloJetJPTNone = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorNone'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (in-cone only)
    RecoCaloJetJPTInCone = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorInCone'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ out-of-cone)
    RecoCaloJetJPTOutOfCone = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorOutOfCone'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ out-of-vertex)
    RecoCaloJetJPTOutOfVertex = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorOutOfVertex'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ pion efficiency)
    RecoCaloJetJPTPionEff = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorPionEff'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ muons)
    RecoCaloJetJPTMuons = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorMuons'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ electrons)
    RecoCaloJetJPTElectrons = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorElectrons'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected reco::CaloJets (+ vectorial using tracks only)
    RecoCaloJetJPTVecTracks = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorVecTracks'),
    leaves = cms.PSet(kine),
    ),
    
    # JPT-corrected reco::CaloJets (+ vectorial using tracks and response)
    RecoCaloJetJPTVecResponse = cms.PSet(
    Class = cms.string('reco::CaloJet'),
    src = cms.InputTag('sort:JPTCorrectorVecResponse'),
    leaves = cms.PSet(kine),
    ),

    # Uncorrected pat::Jets
    PatJetIC5 = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:RawPATJets'),
    leaves = cms.PSet(kine),
    ),

    # ZSP-corrected pat::Jets
    PatJetZSP = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:PATZSPJetCorJetIcone5'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected pat::Jets (old JPT)
    PatJetJPTOld = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:PATJetPlusTrackZSPCorJetIcone5'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected pat::Jets (new JPT)
    PatJetJPT = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:PATJPTCorrectionIC5'),
    leaves = cms.PSet(kine),
    ),

    # JPT-corrected pat::Jets (JTA on-the-fly)
    PatJetJTA = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:JTAJPTCorrectionIC5'),
    leaves = cms.PSet(kine),
    ),
    
    # MC-corrected pat::Jets
    PatJetMC = cms.PSet(
    Class = cms.string('pat::Jet'),
    src = cms.InputTag('sort:cleanLayer1Jets'),
    leaves = cms.PSet(kine),
    ),

    ),
    ),
    )
