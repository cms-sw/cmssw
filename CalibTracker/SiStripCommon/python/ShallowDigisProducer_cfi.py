import FWCore.ParameterSet.Config as cms

shallowDigis = cms.EDProducer("ShallowDigisProducer",
                              DigiProducersList = cms.VInputTag(
    cms.InputTag('siStripDigis','ZeroSuppressed'),
    cms.InputTag('siStripZeroSuppression','VirginRaw'),
    cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
    cms.InputTag('siStripZeroSuppression','ScopeMode'))
                              )

# foo bar baz
# r0DuY9pSxg31a
# WTuaKbX35X5SP
