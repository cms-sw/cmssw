import FWCore.ParameterSet.Config as cms

# This modifier skips the HCAL packer and runs a fake unpacker so all expected collections are still available

hcalSkipPacker = cms.Modifier()
