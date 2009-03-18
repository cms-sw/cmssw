import FWCore.ParameterSet.Config as cms

L1CondDBIOVWriter = cms.EDAnalyzer("L1CondDBIOVWriter",
                                   toPut = cms.VPSet(),
                                   tscKey = cms.string('dummy'),
                                   ignoreTriggerKey = cms.bool(False)
                                   )

