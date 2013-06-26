import FWCore.ParameterSet.Config as cms

DTFakeVDriftESProducer = cms.ESSource("DTFakeVDriftESProducer",
    vDrift = cms.double(0.00543),
    reso = cms.double(0.05)
)


