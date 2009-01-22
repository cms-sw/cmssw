import FWCore.ParameterSet.Config as cms

SiPixelFakeTemplateDBObjectESSource = cms.ESSource("SiPixelFakeTemplateDBObjectESSource",
    siPixelTemplateCalibrations = cms.vstring(
    "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp0001.out",
    "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp0004.out",
    "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp0010.out",
    "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp0012.out"),
    Version = cms.double(1.2)
)


