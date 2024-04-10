import FWCore.ParameterSet.Config as cms

etSumZdcAnalyzer = cms.EDAnalyzer('L1TZDCAnalyzer',
                                  etSumTag = cms.InputTag("etSumZdcProducer")
                                  )
