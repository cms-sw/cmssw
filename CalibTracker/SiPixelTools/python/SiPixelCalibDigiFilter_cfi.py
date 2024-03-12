import FWCore.ParameterSet.Config as cms

siPixelCalibDigiFilter = cms.EDFilter("SiPixelCalibDigiFilter",
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigiProducer")
)

siPixelCalibDigiFilterPath = cms.Path(siPixelCalibDigiFilter)

# foo bar baz
# 52C8Y8yHhl1VT
# gsl8WOpftg5Cx
