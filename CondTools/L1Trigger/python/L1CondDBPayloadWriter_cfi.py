import FWCore.ParameterSet.Config as cms

L1CondDBPayloadWriter = cms.EDAnalyzer("L1CondDBPayloadWriter",
                                       writeL1TriggerKey = cms.bool(True),
                                       writeConfigData = cms.bool(True),
                                       overwriteKeys = cms.bool(False),
                                       logTransactions = cms.bool(False)
                                       )


