import FWCore.ParameterSet.Config as cms

siStripDetInfoFileWriter = cms.EDAnalyzer("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('myfile.txt')
)


