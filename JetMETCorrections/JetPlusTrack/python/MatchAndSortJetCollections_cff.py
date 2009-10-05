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
    genJets = cms.InputTag("cleanLayer1Jets"),
    caloJets = cms.VInputTag(
    "iterativeCone5CaloJets",
    "ZSPJetCorJetIcone5",
    "JetPlusTrackZSPCorJetIcone5",
    "JPTCorrectorDefault",
    "JPTCorrectorNone",
    "JPTCorrectorInCone",
    "JPTCorrectorOutOfCone",
    "JPTCorrectorOutOfVertex",
    "JPTCorrectorPionEff",
    "JPTCorrectorMuons",
    "JPTCorrectorElectrons",
    "JPTCorrectorVecTracks",
    "JPTCorrectorVecResponse",
    ),
    patJets = cms.VInputTag(
    "RawPATJets",
    "PATZSPJetCorJetIcone5",
    "PATJetPlusTrackZSPCorJetIcone5",
    "PATJPTCorrectionIC5",
    "JTAJPTCorrectionIC5",
    "cleanLayer1Jets",
    ),
    )

matchAndSortJetCollections = cms.Sequence(
    #matchToGenJets *
    sortByGenJetPt
    )
