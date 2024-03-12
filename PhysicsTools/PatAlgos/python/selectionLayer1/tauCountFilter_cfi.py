import FWCore.ParameterSet.Config as cms

# module to filter on the number of Taus
countPatTaus = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(0),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("cleanPatTaus")
)


# foo bar baz
# p2lwZYnRWG9sx
