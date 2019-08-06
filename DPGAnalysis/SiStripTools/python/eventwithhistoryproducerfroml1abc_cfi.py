import FWCore.ParameterSet.Config as cms


import EventFilter.Utilities.tcdsRawToDigi_cfi
unpackTcds = EventFilter.Utilities.tcdsRawToDigi_cfi.tcdsRawToDigi.clone(
InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

consecutiveHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                l1ABCCollection=cms.InputTag("scalersRawToDigi"),
                                tcdsRecordLabel= cms.InputTag("unpackTcds","tcdsRecord")
                                )


#consecutiveHEs = cms.Sequence(unpackTcds*task_consecutiveHEs)

