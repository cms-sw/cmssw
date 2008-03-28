import FWCore.ParameterSet.Config as cms

FEDRawDataAnalyzer = cms.EDAnalyzer("SiStripFEDRawDataAnalyzer",
    ProductLabel = cms.untracked.string('source'),
    ProductInstance = cms.untracked.string('')
)


