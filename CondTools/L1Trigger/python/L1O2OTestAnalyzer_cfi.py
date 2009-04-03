import FWCore.ParameterSet.Config as cms

L1O2OTestAnalyzer = cms.EDAnalyzer("L1O2OTestAnalyzer",
                                   printL1TriggerKey = cms.bool(True),
                                   printL1TriggerKeyList = cms.bool(True),
                                   printPayloadTokens = cms.bool(True)
                                  )

