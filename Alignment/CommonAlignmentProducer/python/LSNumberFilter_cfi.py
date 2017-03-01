import FWCore.ParameterSet.Config as cms

lsNumberFilter = cms.EDFilter("LSNumberFilter",
                              minLS = cms.untracked.uint32(21)
                              )
