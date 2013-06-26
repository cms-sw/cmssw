import FWCore.ParameterSet.Config as cms

DTFakeTTrigESProducer = cms.ESSource("DTFakeTTrigESProducer",
    tMean = cms.double(499.609),
    sigma = cms.double(0.0),
    kFactor = cms.double(0.0)
)


