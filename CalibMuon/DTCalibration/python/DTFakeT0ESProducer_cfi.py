import FWCore.ParameterSet.Config as cms

DTFakeT0ESProducer = cms.ESSource("DTFakeT0ESProducer",
    t0Mean = cms.double(0.0),
    t0Sigma = cms.double(0.0)
)



