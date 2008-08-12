import FWCore.ParameterSet.Config as cms

dedxHarmonic2 = cms.EDProducer("DeDxEstimatorProducer",
    trackDeDxHits = cms.InputTag("dedxHits"),
    exponent = cms.double(-2.0),
    estimator = cms.string('generic')
)


