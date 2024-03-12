import FWCore.ParameterSet.Config as cms

L1O2OTestAnalyzerExt = cms.EDAnalyzer("L1O2OTestAnalyzerExt",
                                   printL1TriggerKeyExt = cms.bool(True),
                                   printL1TriggerKeyListExt = cms.bool(True),
                                   printPayloadTokens = cms.bool(True),
                                   printESRecords = cms.bool(True),
                                   recordsToPrint = cms.vstring()
#    'L1TUtmTriggerMenuRcd' ) # Run Settings records
                                  )
# foo bar baz
# w5MfgQ3ACKbOW
# 7rISOnfg2J0Al
