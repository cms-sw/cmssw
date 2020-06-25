import FWCore.ParameterSet.Config as cms

# Import from the generic cfi file for CSC unpacking
import EventFilter.CSCRawToDigi.muonCSCDCCUnpacker_cfi

muonCSCDigis = EventFilter.CSCRawToDigi.muonCSCDCCUnpacker_cfi.muonCSCDCCUnpacker.clone()
# Define input to the unpacker
muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
# Use CSC examiner to check for corrupt or semi-corrupt data & avoid unpacker crashes
muonCSCDigis.UseExaminer = cms.bool(True)
# This mask is needed by the examiner
muonCSCDigis.ExaminerMask = cms.uint32(0x1FEBF7F6)
# Use Examiner to unpack good chambers and skip only bad ones
muonCSCDigis.UseSelectiveUnpacking = cms.bool(True)
# This mask simply reduces error reporting
muonCSCDigis.ErrorMask = cms.uint32(0x0)
# Unpack general status digis?
muonCSCDigis.UnpackStatusDigis = cms.bool(False)
# Unpack FormatStatus digi?
muonCSCDigis.UseFormatStatus = cms.bool(True)
# Turn on lots of output
muonCSCDigis.Debug = cms.untracked.bool(False)
muonCSCDigis.PrintEventNumber = cms.untracked.bool(False)
# Visualization of raw data in corrupted events
muonCSCDigis.VisualFEDInspect = cms.untracked.bool(False)
muonCSCDigis.VisualFEDShort = cms.untracked.bool(False)
muonCSCDigis.FormatedEventDump = cms.untracked.bool(False)

## in Run-3 include GEMs
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( muonCSCDigis, useGEMs = cms.bool(False) )
