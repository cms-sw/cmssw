import FWCore.ParameterSet.Config as cms

CalibrationTrackSelector = cms.EDFilter("CalibrationTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-2.6),
    etaMax = cms.double(2.6),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
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


