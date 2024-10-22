import FWCore.ParameterSet.Config as cms

consecutiveHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                l1ABCCollection=cms.InputTag("scalersRawToDigi"),
                                tcdsRecordLabel= cms.InputTag("tcdsDigis","tcdsRecord"),
                                forceSCAL = cms.bool(True)
                                )
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(consecutiveHEs, forceSCAL = False)
