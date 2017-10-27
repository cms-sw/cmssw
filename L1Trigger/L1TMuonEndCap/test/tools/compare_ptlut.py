import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.analyzer1 = cms.EDAnalyzer("ComparePtLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Input files
    infile1 = cms.string(""),
    infile2 = cms.string(""),
)

import os
infile1 = os.environ.get("CMSSW_BASE") + "/"
infile1 += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut/LUT_AndrewFix_25July16.dat"
infile2 = os.environ.get("CMSSW_BASE") + "/"
#infile2 += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_madorsky/LUT_AndrewFix_25July16.dat"
#infile2 += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest/LUT_AndrewFix_25July16.dat"
#infile2 += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_AndrewFix_25July16.dat"
infile2 += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_v5_24Oct16.dat"
process.analyzer1.infile1 = infile1
process.analyzer1.infile2 = infile2

process.path1 = cms.Path(process.analyzer1)
