import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource_CTPPS = cms.ESProducer("XMLIdealGeometryESProducer",
                                                rootDDName = cms.string('cms:CMSE'),
                                                label = cms.string('CTPPS'),
                                                appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
                                                )

ctppsGeometryESModule = cms.ESProducer("PPSGeometryESProducer",
    verbosity = cms.untracked.uint32(1),
    detectorTag = cms.string("CMS")
)
