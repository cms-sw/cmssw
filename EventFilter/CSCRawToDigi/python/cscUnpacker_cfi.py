import FWCore.ParameterSet.Config as cms

# This is a generic cfi file for CSC unpacking
# tumanov@rice.edu 3/16/07
muonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    # Use CSC examiner for corrupt or semi-corrupt data to avoid unpacker crashes
    UseExaminer = cms.untracked.bool(True),
    # This mask simply reduces error reporting
    ErrorMask = cms.untracked.uint32(3754946559),
    # Define input to the unpacker
    #InputTag InputObjects = cscpacker:CSCRawData
    InputObjects = cms.InputTag("source"),
    # This mask is needed by the examiner if it's used
    ExaminerMask = cms.untracked.uint32(374076406),
    #this flag disables unpacking of the Status Digis
    UnpackStatusDigis = cms.untracked.bool(False),
    #set this to true if unpacking MTCC data from summer-fall MTCC2006 
    isMTCCData = cms.untracked.bool(False),
    Debug = cms.untracked.bool(False)
)


