import FWCore.ParameterSet.Config as cms
# mode:
# 1 - bottom only
# 2 - top only
# 3 - top or bottom
# 4 - top and bottom
CruzetL1DTfilter = cms.EDFilter("CruzetL1DTfilter",
  mode = cms.int32(3), 
  GMTInputTag = cms.InputTag("gtDigis")
)
# foo bar baz
# ByRmjLq0WYZBr
# b3yAQ2t7NocuT
