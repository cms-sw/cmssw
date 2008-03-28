import FWCore.ParameterSet.Config as cms

# Standard Midpoint Cone Jets parameters
# $Id
MconeJetParameters = cms.PSet(
    maxIterations = cms.int32(100),
    coneAreaFraction = cms.double(1.0),
    overlapThreshold = cms.double(0.75),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    maxPairSize = cms.int32(2)
)

