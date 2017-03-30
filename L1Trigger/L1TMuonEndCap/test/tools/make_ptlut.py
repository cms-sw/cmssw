import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.analyzer1 = cms.EDAnalyzer("MakePtLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Versioning
    PtLUTVersion = cms.int32(5),

    # Sector processor pt-assignment parameters
    spPAParams16 = cms.PSet(
        BDTXMLDir       = cms.string('v_16_02_21'),
        ReadPtLUTFile   = cms.bool(False),
        FixMode15HighPt = cms.bool(True),
        Bug9BitDPhi     = cms.bool(False),
        BugMode7CLCT    = cms.bool(False),
        BugNegPt        = cms.bool(False),
    ),

    # Output file
    outfile = cms.string(""),

    # Check addresses
    onlyCheck = cms.bool(False),
    addressesToCheck = cms.vuint64(),
)

import os
outfile = os.environ.get("CMSSW_BASE") + "/"
#outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest/LUT_AndrewFix_25July16.dat"
#outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_AndrewFix_25July16.dat"
outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_v5_24Oct16.dat"
process.analyzer1.outfile = outfile  # make sure the directory exists

#process.analyzer1.spPAParams16.ReadPtLUTFile = True
#process.analyzer1.onlyCheck = True
process.analyzer1.addressesToCheck = [
    ((939524096+0) | (0<<18)),
    ((939524096+1) | (0<<18)),
    ((939524096+2) | (0<<18)),
    ((939524096+3) | (0<<18)),
    ((939524096+4) | (0<<18)),
    ((939524096+5) | (0<<18)),
    ((939524096+6) | (0<<18)),
    ((939524096+7) | (0<<18)),
    ((939524096+8) | (0<<18)),
    ((939524096+9) | (0<<18)),

    ((939524096+0) | (1<<18)),
    ((939524096+1) | (1<<18)),
    ((939524096+2) | (1<<18)),
    ((939524096+3) | (1<<18)),
    ((939524096+4) | (1<<18)),
    ((939524096+5) | (1<<18)),
    ((939524096+6) | (1<<18)),
    ((939524096+7) | (1<<18)),
    ((939524096+8) | (1<<18)),
    ((939524096+9) | (1<<18)),

    ((939524096+0) | (2<<18)),
    ((939524096+1) | (2<<18)),
    ((939524096+2) | (2<<18)),
    ((939524096+3) | (2<<18)),
    ((939524096+4) | (2<<18)),
    ((939524096+5) | (2<<18)),
    ((939524096+6) | (2<<18)),
    ((939524096+7) | (2<<18)),
    ((939524096+8) | (2<<18)),
    ((939524096+9) | (2<<18)),

    ((939524096+0) | (3<<18)),
    ((939524096+1) | (3<<18)),
    ((939524096+2) | (3<<18)),
    ((939524096+3) | (3<<18)),
    ((939524096+4) | (3<<18)),
    ((939524096+5) | (3<<18)),
    ((939524096+6) | (3<<18)),
    ((939524096+7) | (3<<18)),
    ((939524096+8) | (3<<18)),
    ((939524096+9) | (3<<18)),

    ((939524096+0) | (4<<18)),
    ((939524096+1) | (4<<18)),
    ((939524096+2) | (4<<18)),
    ((939524096+3) | (4<<18)),
    ((939524096+4) | (4<<18)),
    ((939524096+5) | (4<<18)),
    ((939524096+6) | (4<<18)),
    ((939524096+7) | (4<<18)),
    ((939524096+8) | (4<<18)),
    ((939524096+9) | (4<<18)),

    ((939524096+0) | (5<<18)),
    ((939524096+1) | (5<<18)),
    ((939524096+2) | (5<<18)),
    ((939524096+3) | (5<<18)),
    ((939524096+4) | (5<<18)),
    ((939524096+5) | (5<<18)),
    ((939524096+6) | (5<<18)),
    ((939524096+7) | (5<<18)),
    ((939524096+8) | (5<<18)),
    ((939524096+9) | (5<<18)),

    ((939524096+0) | (6<<18)),
    ((939524096+1) | (6<<18)),
    ((939524096+2) | (6<<18)),
    ((939524096+3) | (6<<18)),
    ((939524096+4) | (6<<18)),
    ((939524096+5) | (6<<18)),
    ((939524096+6) | (6<<18)),
    ((939524096+7) | (6<<18)),
    ((939524096+8) | (6<<18)),
    ((939524096+9) | (6<<18)),

    ((939524096+0) | (7<<18)),
    ((939524096+1) | (7<<18)),
    ((939524096+2) | (7<<18)),
    ((939524096+3) | (7<<18)),
    ((939524096+4) | (7<<18)),
    ((939524096+5) | (7<<18)),
    ((939524096+6) | (7<<18)),
    ((939524096+7) | (7<<18)),
    ((939524096+8) | (7<<18)),
    ((939524096+9) | (7<<18)),

]


process.path1 = cms.Path(process.analyzer1)
