import FWCore.ParameterSet.Config as cms

L1CondDBIOVWriter = cms.EDAnalyzer("L1CondDBIOVWriter",
                                   toPut = cms.VPSet(),
                                   tscKey = cms.string('dummy'),
                                   ignoreTriggerKey = cms.bool(False),
                                   logKeys = cms.bool(False),
                                   logTransactions = cms.bool(False),
                                   forceUpdate = cms.bool(False)
                                   )

# foo bar baz
# NmfOVy1HuJRJk
# KkwR3aucJ0w3j
