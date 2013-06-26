import FWCore.ParameterSet.Config as cms

SingleTrajectoryUpdatorForPixels = cms.PSet(
    AlignmentUpdatorName = cms.string( "SingleTrajectoryUpdator" ),

    MinNumberOfHits = cms.uint32(1),
    ExtraWeight = cms.double(1e-06),
    ExternalPredictionWeight = cms.double(10.0),
    CheckCovariance = cms.bool( False ),
    NumberOfPreAlignmentEvts = cms.uint32(0)
)

SingleTrajectoryUpdatorForStrips = cms.PSet(
    AlignmentUpdatorName = cms.string( "SingleTrajectoryUpdator" ),

    MinNumberOfHits = cms.uint32(1),
    ExtraWeight = cms.double(0.0001),
    ExternalPredictionWeight = cms.double(10.0),
    CheckCovariance = cms.bool( False ),
    NumberOfPreAlignmentEvts = cms.uint32(0)
)

DummyUpdator = cms.PSet(
    AlignmentUpdatorName = cms.string( "DummyUpdator" )
)

