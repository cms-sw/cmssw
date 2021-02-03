import FWCore.ParameterSet.Config as cms

hcalDDDSimulationConstants = cms.ESProducer("HcalDDDSimulationConstantsESModule",
    appendToDataLabel = cms.string('')
)
