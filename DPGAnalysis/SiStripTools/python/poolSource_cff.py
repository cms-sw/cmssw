import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

# foo bar baz
# DrQVz5GB73GgF
# gksw40UtUAgsR
