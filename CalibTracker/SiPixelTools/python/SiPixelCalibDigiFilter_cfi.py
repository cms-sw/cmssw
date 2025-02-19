import FWCore.ParameterSet.Config as cms

siPixelCalibDigiFilter = cms.EDFilter("SiPixelCalibDigiFilter",
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigiProducer")
)

siPixelCalibDigiFilterPath = cms.Path(siPixelCalibDigiFilter)

