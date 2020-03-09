import FWCore.ParameterSet.Config as cms

import EventFilter.Utilities.tcdsRawToDigi_cfi
from Configuration.StandardSequences.SimL1EmulatorRepack_Full_cff import unpackTcds

consecutiveHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                l1ABCCollection=cms.InputTag("scalersRawToDigi"),
                                tcdsRecordLabel= cms.InputTag("unpackTcds","tcdsRecord"),
                                forceSCAL = cms.bool(True)
                                )
