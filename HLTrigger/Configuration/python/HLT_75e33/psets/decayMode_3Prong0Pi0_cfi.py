import FWCore.ParameterSet.Config as cms

decayMode_3Prong0Pi0 = cms.PSet(
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        mass = cms.bool(False),
        phi = cms.bool(False)
    ),
    assumeStripMass = cms.double(-1.0),
    maxMass = cms.string('1.5'),
    maxPi0Mass = cms.double(1000000000.0),
    minMass = cms.double(0.8),
    minPi0Mass = cms.double(-1000.0),
    nCharged = cms.uint32(3),
    nChargedPFCandsMin = cms.uint32(1),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2)
)