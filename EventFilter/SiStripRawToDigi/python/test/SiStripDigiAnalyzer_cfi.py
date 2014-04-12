import FWCore.ParameterSet.Config as cms

DigiAnalyzer = cms.EDAnalyzer("SiStripDigiAnalyzer",
    InputModuleLabel = cms.string('siStripDigis')
)


