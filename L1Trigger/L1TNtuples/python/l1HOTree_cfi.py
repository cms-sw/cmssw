import FWCore.ParameterSet.Config as cms

l1HOTree = cms.EDAnalyzer(
    "L1HOTreeProducer",
    hoDataFrameToken = cms.untracked.InputTag("hcalDigis")
)
# foo bar baz
# bxbKDaadgN3tL
# vE6dgPqnX77oY
