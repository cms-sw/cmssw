import FWCore.ParameterSet.Config as cms

decayMode_1Prong0Pi0 = cms.PSet(
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        mass = cms.bool(True),
        phi = cms.bool(True)
    ),
    assumeStripMass = cms.double(-1.0),
    maxMass = cms.string('1.'),
    maxPi0Mass = cms.double(1000000000.0),
    minMass = cms.double(-1000.0),
    minPi0Mass = cms.double(-1000.0),
    nCharged = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(1)
)