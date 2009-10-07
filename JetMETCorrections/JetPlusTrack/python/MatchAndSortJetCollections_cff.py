import FWCore.ParameterSet.Config as cms

matchToGenJets = cms.EDFilter(
    "GenJetMatcher",
    src = cms.InputTag("iterativeCone5CaloJets"),
    matched = cms.InputTag("iterativeCone5GenJets"),
    mcPdgId = cms.vint32(),  # Not used
    mcStatus = cms.vint32(), # Not used
    checkCharge = cms.bool(False),
    maxDeltaR = cms.double(0.4),
    maxDPtRel = cms.double(3.0),
    resolveAmbiguities = cms.bool(True),     # True: Forbids two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False), # False: just matches input in order (True: picks lowest deltaR pair first)
    )

sortByGenJetPt = cms.EDProducer(
    "SortJetCollectionsByGenJetPt",
    src     = cms.InputTag("iterativeCone5CaloJets"),
    matched = cms.InputTag("matchToGenJets"),
    jets    = cms.VInputTag(
    "iterativeCone5CaloJets",
    "ZSPJetCorJetIcone5",
    "JPTCorJetIC5CaloDefault",
    "JPTCorJetIC5CaloNone",
    "JPTCorJetIC5CaloInCone",
    "JPTCorJetIC5CaloOutOfCone",
    "JPTCorJetIC5CaloOutOfVertex",
    "JPTCorJetIC5CaloPionEff",
    "JPTCorJetIC5CaloMuons",
    "JPTCorJetIC5CaloElectrons",
    "JPTCorJetIC5CaloVecTracks",
    "JPTCorJetIC5CaloVecResponse",
    "uncorrectedLayer1JetsIC5",
    "PatZSPCorJetIC5Calo",
    "PatJPTCorJetIC5Calo",
    "selectedLayer1Jets",
    ),
    )

matchAndSortJetCollections = cms.Sequence(
    matchToGenJets *
    sortByGenJetPt
    )
