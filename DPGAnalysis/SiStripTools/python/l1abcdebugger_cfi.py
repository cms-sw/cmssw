import FWCore.ParameterSet.Config as cms

l1abcdebugger = cms.EDAnalyzer("L1ABCDebugger",
                               l1ABCCollection=cms.InputTag("scalersRawToDigi")
                               )
