import FWCore.ParameterSet.Config as cms

lsNumberFilter = cms.EDFilter("LSNumberFilter",
                              minLS = cms.untracked.uint32(21)
                              )
# foo bar baz
