import FWCore.ParameterSet.Config as cms

SingleTrajectoryUpdatorForPixels = cms.PSet(
    CheckCovariance = cms.untracked.bool(False),
    ExtraWeight = cms.untracked.double(1e-06),
    ExternalPredictionWeight = cms.untracked.double(10.0),
    AlignmentUpdatorName = cms.string('SingleTrajectoryUpdator')
)
SingleTrajectoryUpdatorForStrips = cms.PSet(
    CheckCovariance = cms.untracked.bool(False),
    ExtraWeight = cms.untracked.double(0.0001),
    ExternalPredictionWeight = cms.untracked.double(10.0),
    AlignmentUpdatorName = cms.string('SingleTrajectoryUpdator')
)
DummyUpdator = cms.PSet(
    AlignmentUpdatorName = cms.string('DummyUpdator')
)

