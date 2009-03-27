import FWCore.ParameterSet.Config as cms

# This is a generic cfi file for CSC unpacking

muonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    # Use CSC examiner for corrupt or semi-corrupt data to avoid unpacker crashes
    UseExaminer = cms.untracked.bool(True),
    # This mask simply reduces error reporting
    ErrorMask = cms.untracked.uint32(0x0),
    # Define input to the unpacker
    #InputTag InputObjects = cscpacker:CSCRawData
    InputObjects = cms.InputTag("source"),
    # This mask is needed by the examiner 
    ExaminerMask = cms.untracked.uint32(0x1FEBF3F6),
    # Unpack general status digis?
    UnpackStatusDigis = cms.untracked.bool(False),
    # Unpack FormatStatus digi?
    UseFormatStatus = cms.untracked.bool(True),                        
    # Use Examiner to unpack good chambers and skip only bad ones
    UseSelectiveUnpacking = cms.untracked.bool(True),
    # Turn on lots of output                            
    Debug = cms.untracked.bool(False)
)


