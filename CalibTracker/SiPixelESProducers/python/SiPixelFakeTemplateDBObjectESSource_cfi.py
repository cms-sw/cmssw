import FWCore.ParameterSet.Config as cms

SiPixelFakeTemplateDBObjectESSource = cms.ESSource("SiPixelFakeTemplateDBObjectESSource",
    siPixelTemplateCalibrations = cms.vstring(
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0001.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0004.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0011.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0012.out"),
    Version = cms.double(1.3)
)


