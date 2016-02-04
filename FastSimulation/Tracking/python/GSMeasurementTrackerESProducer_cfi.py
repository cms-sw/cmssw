import FWCore.ParameterSet.Config as cms

MeasurementTrackerESProducer = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string(''),
    GaussianSmearing = cms.bool(True),
    PixelCPE = cms.string('Fake'),
    HitMatcher = cms.string('Fake')
)


