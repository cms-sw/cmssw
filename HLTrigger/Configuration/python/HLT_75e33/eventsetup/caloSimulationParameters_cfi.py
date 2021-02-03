import FWCore.ParameterSet.Config as cms

caloSimulationParameters = cms.ESProducer("CaloSimParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False)
)
