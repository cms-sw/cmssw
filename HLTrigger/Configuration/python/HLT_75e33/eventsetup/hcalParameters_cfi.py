import FWCore.ParameterSet.Config as cms

hcalParameters = cms.ESProducer("HcalParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False)
)
