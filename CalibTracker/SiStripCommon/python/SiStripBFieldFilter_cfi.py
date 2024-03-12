import FWCore.ParameterSet.Config as cms

siStripBFieldOnFilter  = cms.EDFilter("SiStripBFieldFilter")

siStripBFieldOffFilter = cms.EDFilter("SiStripBFieldFilter",
                                      HIpassFilter = cms.untracked.bool(False)
                                     )
# foo bar baz
# cXjZhv2de8DLJ
