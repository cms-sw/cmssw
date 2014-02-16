import FWCore.ParameterSet.Config as cms

localMaxSeeds_EB = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    ### seed finding parameters
    seedingThreshold = cms.double(0.23),
    seedingThresholdPt = cms.double(0.0),
    nNeighbours = cms.uint32(8)
    )

localMaxSeeds_EE = localMaxSeeds_EB.clone(
    seedingThreshold = cms.double(0.6),
    seedingThresholdPt = cms.double(0.15)
    )

localMaxSeeds_PS = localMaxSeeds_EB.clone(
    seedingThreshold = cms.double(1.2e-4),
    nNeighbours = cms.uint32(4)
    )

