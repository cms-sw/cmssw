import FWCore.ParameterSet.Config as cms

# producer for Hcal Zmumu (HCAL Zmumu for ho)
# copied from Zmumu alignment
ALCARECOHcalCalZMuMu = cms.EDFilter("AlignmentMuonSelectorModule",
    chi2nMaxSA = cms.double(9999999.0),
    nHitMaxTO = cms.double(9999999.0),
    nHitMaxGB = cms.double(9999999.0),
    applyMultiplicityFilter = cms.bool(False),
    etaMin = cms.double(-2.4),
    minMassPair = cms.double(89.0),
    etaMax = cms.double(2.4),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    ptMin = cms.double(0.0),
    minMultiplicity = cms.int32(1),
    applyNHighestPt = cms.bool(False),
    nHitMaxSA = cms.double(9999999.0),
    ptMax = cms.double(999.0),
    # Stand Alone Muons
    nHitMinSA = cms.double(0.0),
    chi2nMaxTO = cms.double(9999999.0),
    chi2nMaxGB = cms.double(9999999.0),
    nHighestPt = cms.int32(2),
    # copy best mass pair combination muons to result vector
    # Criteria: 
    # a) maxMassPair != minMassPair: the two highest pt muons with mass pair inside the given mass window
    # b) maxMassPair == minMassPair: the muon pair with mass pair closest to given mass value
    applyMassPairFilter = cms.bool(False),
    src = cms.InputTag("muons"), ## globalMuons

    # Global Muons
    nHitMinGB = cms.double(0.0),
    filter = cms.bool(True),
    maxMassPair = cms.double(90.0),
    # Tracker Only
    nHitMinTO = cms.double(0.0),
    applyBasicCuts = cms.bool(True)
)


