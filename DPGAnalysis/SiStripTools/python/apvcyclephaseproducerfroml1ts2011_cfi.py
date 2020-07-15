import FWCore.ParameterSet.Config as cms

APVPhases = cms.EDProducer('APVCyclePhaseProducerFromL1TS',
                           ignoreDB = cms.untracked.bool(True),
                           defaultPartitionNames = cms.vstring("TI",
                                                               "TO",
                                                               "TP",
                                                               "TM"
                                                               ),
                           defaultPhases = cms.vint32(60,60,60,60),
                           magicOffset = cms.untracked.int32(258),
                           l1TSCollection = cms.InputTag("scalersRawToDigi"),
                           tcdsRecordLabel= cms.InputTag("tcdsDigis","tcdsRecord"),
                           forceSCAL = cms.bool(True),
                           )

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(APVPhases, forceSCAL = False)
