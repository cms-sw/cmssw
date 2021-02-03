import FWCore.ParameterSet.Config as cms

decayMode_1Prong2Pi0 = cms.PSet(
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        mass = cms.bool(True),
        phi = cms.bool(True)
    ),
    assumeStripMass = cms.double(0.0),
    maxMass = cms.string('max(1.2, min(1.2*sqrt(pt/100.), 4.0))'),
    maxPi0Mass = cms.double(0.2),
    minMass = cms.double(0.4),
    minPi0Mass = cms.double(0.05),
    nCharged = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    nPiZeros = cms.uint32(2),
    nTracksMin = cms.uint32(1)
)