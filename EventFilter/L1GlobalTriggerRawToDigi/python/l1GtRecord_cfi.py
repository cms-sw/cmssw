import FWCore.ParameterSet.Config as cms

l1GtRecord = cms.EDProducer("L1GlobalTriggerRecordProducer",
    # InputTag for the L1 Global Trigger DAQ readout record
    #   GT Emulator = gtDigis
    #   GT Unpacker = gtDigis
    #
    L1GtReadoutRecordTag = cms.InputTag("gtDigis")
)


