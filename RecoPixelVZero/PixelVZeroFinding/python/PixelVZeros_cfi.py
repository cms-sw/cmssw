import FWCore.ParameterSet.Config as cms

pixelVZeros = cms.EDProducer("PixelVZeroProducer",
    maxCrossingRadius = cms.double(5.0),
    maxImpactMother = cms.double(0.2),
    maxDcaR = cms.double(0.2),
    minImpactNegativeDaughter = cms.double(0.0),
    maxDcaZ = cms.double(0.2),
    minImpactPositiveDaughter = cms.double(0.0),
    minCrossingRadius = cms.double(0.5)
)


