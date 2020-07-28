import FWCore.ParameterSet.Config as cms

muonGeometryConstants = cms.ESProducer("MuonGeometryConstantsESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False)
)
