import FWCore.ParameterSet.Config as cms

shallowTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      )
   )
