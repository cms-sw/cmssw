import FWCore.ParameterSet.Config as cms

SiPixelFakeGenErrorDBObjectESSource = cms.ESSource("SiPixelFakeGenErrorDBObjectESSource",
    siPixelGenErrorCalibrations = cms.vstring(
    "CalibTracker/SiPixelESProducers/data/generror_summary_zp0030.out",
    "CalibTracker/SiPixelESProducers/data/generror_summary_zp0031.out"),
    Version = cms.double(1.3)
)


