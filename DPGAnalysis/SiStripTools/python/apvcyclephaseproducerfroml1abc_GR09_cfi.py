import FWCore.ParameterSet.Config as cms

APVPhases = cms.EDProducer('APVCyclePhaseProducerFromL1ABC',
                           defaultPartitionNames = cms.vstring("TI",
                                                               "TO",
                                                               "TP",
                                                               "TM"
                                                               ),
                           defaultPhases = cms.vint32(30,30,30,30),
                           StartOfRunOrbitOffset = cms.int32(4),
                           l1ABCCollection = cms.InputTag("scalersRawToDigi"),
                           )

