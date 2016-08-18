import FWCore.ParameterSet.Config as cms

L1CondDBPayloadWriterExt = cms.EDAnalyzer("L1CondDBPayloadWriterExt",
                                       writeL1TriggerKeyExt = cms.bool(True),
                                       writeConfigData = cms.bool(True),
                                       overwriteKeys = cms.bool(False),
                                       logTransactions = cms.bool(False),
                                       newL1TriggerKeyListExt = cms.bool(False)
                                       )


