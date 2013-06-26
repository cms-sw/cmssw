import FWCore.ParameterSet.Config as cms

AlignmentMuonSelector = cms.EDFilter("AlignmentMuonSelectorModule",
    src = cms.InputTag("muons"),
    filter = cms.bool(True),

    applyBasicCuts = cms.bool(True),

    pMin = cms.double(0.0),
    pMax = cms.double(999999.0),
    ptMin = cms.double(10.0),
    ptMax = cms.double(999999.0),
    etaMin = cms.double(-2.4),
    etaMax = cms.double(2.4),
    phiMin = cms.double(-3.1416),
    phiMax = cms.double(3.1416),

    # Stand Alone Muons
    nHitMinSA = cms.double(0.0),
    nHitMaxSA = cms.double(9999999.0),
    chi2nMaxSA = cms.double(9999999.0),

    # Global Muons
    nHitMinGB = cms.double(0.0),
    nHitMaxGB = cms.double(9999999.0),
    chi2nMaxGB = cms.double(9999999.0),

    # Tracker Only
    nHitMinTO = cms.double(0.0),
    nHitMaxTO = cms.double(9999999.0),
    chi2nMaxTO = cms.double(9999999.0),

    applyNHighestPt = cms.bool(False),
    nHighestPt = cms.int32(2),

    applyMultiplicityFilter = cms.bool(False),
    minMultiplicity = cms.int32(1),

    # copy best mass pair combination muons to result vector
    # Criteria: 
    # a) maxMassPair != minMassPair: the two highest pt muons with mass pair inside the given mass window
    # b) maxMassPair == minMassPair: the muon pair with mass pair closest to given mass value
    applyMassPairFilter = cms.bool(False),
    minMassPair = cms.double(89.0),
    maxMassPair = cms.double(90.0)
)
