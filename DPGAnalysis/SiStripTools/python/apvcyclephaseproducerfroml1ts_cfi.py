import FWCore.ParameterSet.Config as cms

APVPhases = cms.EDProducer('APVCyclePhaseProducerFromL1TS',
                           ignoreDB = cms.untracked.bool(True),
                           defaultPartitionNames = cms.vstring("TI",
                                                               "TO",
                                                               "TP",
                                                               "TM"
                                                               ),
                           defaultPhases = cms.vint32(30,30,30,30),
                           l1TSCollection = cms.InputTag("scalersRawToDigi"),
                           )

