import FWCore.ParameterSet.Config as cms

decayMode_1Prong1Pi0 = cms.PSet(
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        mass = cms.bool(True),
        phi = cms.bool(True)
    ),
    assumeStripMass = cms.double(0.1349),
    maxMass = cms.string('max(1.3, min(1.3*sqrt(pt/100.), 4.2))'),
    maxPi0Mass = cms.double(1000000000.0),
    minMass = cms.double(0.3),
    minPi0Mass = cms.double(-1000.0),
    nCharged = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(1)
)