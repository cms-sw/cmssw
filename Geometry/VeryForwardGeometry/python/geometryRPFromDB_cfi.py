import FWCore.ParameterSet.Config as cms
from Geometry.VeryForwardGeometryBuilder.CTPPSGeometryESModule_cfi import CTPPSGeometryESModule

XMLIdealGeometryESSource_CTPPS = cms.ESProducer("XMLIdealGeometryESProducer",
                                                rootDDName = cms.string('cms:CMSE'),
                                                label = cms.string('CTPPS'),
                                                appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
                                                )

ctppsGeometryESModule = CTPPSGeometryESModule.clone(
    compactViewTag = cms.string('XMLIdealGeometryESSource_CTPPS')
)

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsGeometryESModule, isRun2=True)
