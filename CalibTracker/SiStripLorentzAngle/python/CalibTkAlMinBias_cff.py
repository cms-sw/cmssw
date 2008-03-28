import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
ALCARECOTkAlMinBias = cms.EDFilter("AlignmentTrackSelectorModule",
    nHighestPt = cms.int32(2),
    src = cms.InputTag("ctfWithMaterialTracks"),
    minMultiplicity = cms.int32(2),
    nHitMax = cms.double(99.0),
    phiMin = cms.double(-3.1416),
    applyMultiplicityFilter = cms.bool(True),
    applyNHighestPt = cms.bool(False),
    phiMax = cms.double(3.1416),
    filter = cms.bool(True),
    chi2nMax = cms.double(999999.0),
    etaMin = cms.double(-2.4),
    maxMultiplicity = cms.int32(999999),
    ptMin = cms.double(0.8),
    minHitsPerSubDet = cms.PSet(
        inTID = cms.int32(0),
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHitMin = cms.double(6.0),
    ptMax = cms.double(999.0),
    etaMax = cms.double(2.4),
    applyBasicCuts = cms.bool(True)
)

seqALCARECOTkAlMinBias = cms.Sequence(ALCARECOTkAlMinBias)

