import FWCore.ParameterSet.Config as cms

l1decsctf = cms.EDAnalyzer("CSCTFDataToEmuComparator",
    outFile = cms.untracked.string(""),
    PTLUT = cms.PSet(
    LowQualityFlag = cms.untracked.uint32(4),
    ReadPtLUT = cms.untracked.bool(False),
    PtMethod = cms.untracked.uint32(1)
    ))
#TODO: please add input tags for interfacing with the hardware sequence
