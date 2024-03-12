import FWCore.ParameterSet.Config as cms

consumer = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('digis.root')
)
# foo bar baz
# MRNQ2HAUovFsO
