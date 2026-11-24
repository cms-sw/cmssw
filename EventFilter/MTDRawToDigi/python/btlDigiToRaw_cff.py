import FWCore.ParameterSet.Config as cms

# readout map (DetId -> electronics ids) needed by the packer
from EventFilter.MTDRawToDigi.BTLReadoutMapESProducer_cfi import BTLReadoutMapESProducer

# BTL packer: BTL digis -> RawDataBuffer (btlDigiToRaw_cfi is generated from the fillDescriptions of BTLDigiToRaw)
from EventFilter.MTDRawToDigi.btlDigiToRaw_cfi import btlDigiToRaw

# with premixing (premix_stage2) the BTL digis come from mixData
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(btlDigiToRaw, btlDigiCollection = "mixData:MTDBarrel")
