import FWCore.ParameterSet.Config as cms

hltMkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
    appendToDataLabel = cms.string('')
)
