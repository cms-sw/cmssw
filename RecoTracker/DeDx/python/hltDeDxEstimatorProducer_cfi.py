import FWCore.ParameterSet.Config as cms

DeDxEstimatorProducer = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("hltIter4Merged"),
    trajectoryTrackAssociation = cms.InputTag("hltIter4Merged"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(False),
)
