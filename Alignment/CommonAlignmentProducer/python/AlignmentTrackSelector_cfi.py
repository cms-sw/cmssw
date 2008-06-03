# The following comments couldn't be translated into the new config version:

#Settings for the Global TrackSelector

import FWCore.ParameterSet.Config as cms

AlignmentTrackSelector = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    pMin = cms.double(0.0),
    etaMin = cms.double(-2.6),
    minHitIsolation = cms.double(0.01),
    etaMax = cms.double(2.6),
    pMax = cms.double(9999.0),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        #for isolation Tests
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minGlobalMuonCount = cms.int32(1),
        minJetDeltaR = cms.double(0.2),
        maxJetCount = cms.int32(3),
        #for global muon finding
        applyGlobalMuonFilter = cms.bool(False),
        minJetPt = cms.double(40.0), ##GeV

        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0), ##GeV

        #for Jet Count
        applyJetCountFilter = cms.bool(False),
        maxTrackDeltaR = cms.double(0.001)
    ),
    # Settings for the two Body Decay TrackSelector
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105), ##GeV

        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0), ##GeV

        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0), ##radian

        minXMass = cms.double(0.0), ##GeV

        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(10.0),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"), ##ctfWithMaterialTracks

    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(False),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    # Settings for the base TrackSelector 	
    # FIXME this should get its own PSet
    applyBasicCuts = cms.bool(True)
)


