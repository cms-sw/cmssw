import FWCore.ParameterSet.Config as cms

patAnalyzerKit = cms.EDFilter("PatAnalyzerKit",
    ntuplize = cms.string('all'),
    outputTextName = cms.string('PatAnalyzerKit_output.txt'),
    enable = cms.string(''),
    disable = cms.string('')
)


