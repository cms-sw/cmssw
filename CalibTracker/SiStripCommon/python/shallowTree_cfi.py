import FWCore.ParameterSet.Config as cms

shallowTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      )
   )
# foo bar baz
# FzaTSoOV8H4T3
# 7xAND9Fb5fJwL
