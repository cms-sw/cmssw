import FWCore.ParameterSet.Config as cms

siStripDetInfoFileWriter = cms.EDAnalyzer("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('myfile.txt')
)


# foo bar baz
# ICd64gi1S31iD
# h0jUT1PnZr9nT
