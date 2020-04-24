import FWCore.ParameterSet.Config as cms

L1CondDBIOVWriterExt = cms.EDAnalyzer("L1CondDBIOVWriterExt",
                                   toPut  = cms.VPSet(),
                                   tscKey = cms.string('dummy'),
                                   rsKey  = cms.string('dummy'),
                                   ignoreTriggerKey = cms.bool(False),
                                   logKeys = cms.bool(False),
                                   logTransactions = cms.bool(False),
                                   forceUpdate = cms.bool(False)
                                   )

