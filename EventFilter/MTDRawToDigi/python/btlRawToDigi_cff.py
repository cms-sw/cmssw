import FWCore.ParameterSet.Config as cms

# readout map and electronics -> DetId mapping needed by the unpacker
from EventFilter.MTDRawToDigi.BTLReadoutMapESProducer_cfi import BTLReadoutMapESProducer
from EventFilter.MTDRawToDigi.BTLElectronicsToDetIdMappingESProducer_cfi import BTLElectronicsToDetIdMappingESProducer

# BTL alpaka unpacker: RawDataBuffer -> BTL digis (SoA)
from EventFilter.MTDRawToDigi.btlRawToDigi_cfi import btlRawToDigi
