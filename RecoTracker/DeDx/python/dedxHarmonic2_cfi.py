import FWCore.ParameterSet.Config as cms

dedxHarmonic2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    MeVperADCStrip = cms.double(3.61e-06*250),
    MeVperADCPixel = cms.double(3.61e-06),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
)
