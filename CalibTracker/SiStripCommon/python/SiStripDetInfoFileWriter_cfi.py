import FWCore.ParameterSet.Config as cms

siStripDetInfoFileWriter = cms.EDFilter("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('myfile.txt')
)


