import FWCore.ParameterSet.Config as cms

ecalSimulationParametersEE = cms.ESProducer("EcalSimParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False),
    name = cms.string('EcalHitsEE')
)
