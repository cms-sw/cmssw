import FWCore.ParameterSet.Config as cms

FEDRawDataAnalyzer = cms.EDAnalyzer("SiStripFEDRawDataAnalyzer",
    InputLabel = cms.InputTag("source")
)


