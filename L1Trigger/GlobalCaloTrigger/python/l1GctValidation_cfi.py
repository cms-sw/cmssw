import FWCore.ParameterSet.Config as cms

l1GctValidation = cms.EDAnalyzer("L1GctValidation",
    rctInputTag  = cms.untracked.InputTag("simRctDigis"),
    gctInputTag  = cms.untracked.InputTag("simGctDigis")
)


# foo bar baz
# JFDkquJAzXfYB
# 8aiPztiAnU0Cl
