import FWCore.ParameterSet.Config as cms

errorSummaryFilter = cms.EDFilter("ErrorSummaryFilter",
                                  src = cms.InputTag("logErrorHarvester"), 
                                  severity = cms.string("error"),
                                  modules = cms.vstring(),
                                  avoidCategories = cms.vstring()
                                  )
# foo bar baz
# i8BTu1rSJuRWY
# 36IvKPVndgegH
