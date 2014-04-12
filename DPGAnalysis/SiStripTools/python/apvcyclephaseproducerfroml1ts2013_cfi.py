import FWCore.ParameterSet.Config as cms

APVPhases = cms.EDProducer('APVCyclePhaseProducerFromL1TS',
                           ignoreDB = cms.untracked.bool(True),
                           defaultPartitionNames = cms.vstring("TI",
                                                               "TO",
                                                               "TP",
                                                               "TM"
                                                               ),
                           defaultPhases = cms.vint32(48,48,48,48),
                           magicOffset = cms.untracked.int32(11),
                           l1TSCollection = cms.InputTag("scalersRawToDigi"),
                           )

