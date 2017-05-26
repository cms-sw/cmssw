import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.analyzer1 = cms.EDAnalyzer("MakePtLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Versioning
    PtLUTVersion = cms.int32(6),

    # Era
    Era = cms.string('Run2_2017'),

    # Sector processor pt-assignment parameters
    spPAParams16 = cms.PSet(
        BDTXMLDir       = cms.string('NULLITY'),  ## Was v_16_02_21
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
#outfile = os.environ.get("CMSSW_BASE") + "/"
#outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest/LUT_AndrewFix_25July16.dat"
#outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_AndrewFix_25July16.dat"
#outfile += "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut_jftest2/LUT_v5_24Oct16.dat"
outfile = "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/LUTs/2017_05_24/LUT_v6_24May17_backup.dat"

process.analyzer1.outfile = outfile  # make sure the directory exists

process.path1 = cms.Path(process.analyzer1)
