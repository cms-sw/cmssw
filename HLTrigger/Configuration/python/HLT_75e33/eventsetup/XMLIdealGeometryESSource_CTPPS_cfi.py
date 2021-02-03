import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource_CTPPS = cms.ESProducer("XMLIdealGeometryESProducer",
    appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS'),
    label = cms.string('CTPPS'),
    rootDDName = cms.string('cms:CMSE')
)
