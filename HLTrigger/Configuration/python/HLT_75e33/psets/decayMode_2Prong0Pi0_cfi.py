import FWCore.ParameterSet.Config as cms

decayMode_2Prong0Pi0 = cms.PSet(
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        mass = cms.bool(False),
        phi = cms.bool(False)
    ),
    assumeStripMass = cms.double(-1.0),
    maxMass = cms.string('1.2'),
    maxPi0Mass = cms.double(1000000000.0),
    minMass = cms.double(0.0),
    minPi0Mass = cms.double(-1000.0),
    nCharged = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2)
)