import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

iVer = 7
iNum = 57
iDen = 64

process.analyzer1 = cms.EDAnalyzer("MakePtLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Versioning
    PtLUTVersion = cms.int32(iVer),
    numerator    = cms.int32(iNum),
    denominator  = cms.int32(iDen),

    # Sector processor pt-assignment parameters
    spPAParams16 = cms.PSet(
        BDTXMLDir       = cms.string('2017_v7'),
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
# out_dir = os.environ.get("CMSSW_BASE") + "/"
out_dir = '/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/LUTs/'

# out_file = out_dir+'2017_05_24/LUT_v6_24May17_part_32_32.dat"
out_file = out_dir+'2017_06_07/LUT_v%02d_07June17_part_%02d_%02d.dat' % ( iVer, iNum, iDen )

process.analyzer1.outfile = out_file  # make sure the directory exists

process.path1 = cms.Path(process.analyzer1)
