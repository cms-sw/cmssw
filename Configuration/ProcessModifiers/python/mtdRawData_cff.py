import FWCore.ParameterSet.Config as cms

# This modifier runs the BTL local reconstruction from the BTL raw data:
# - BTL packer (BTL digis -> raw data) in DIGI2RAW, with the raw data kept in the RAW event content
# - BTL alpaka unpacker (raw data -> BTL digis SoA) in RAW2DIGI
# - BTL SoA local reconstruction reading the digis of the unpacker
mtdRawData = cms.Modifier()
