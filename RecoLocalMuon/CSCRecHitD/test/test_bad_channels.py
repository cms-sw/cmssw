## Run standalone CSCRecoBadChannelsAnalyzer - test bad strip channels - Tim Cox - 07.10.2014
## This version runs in 720pre6 on a real data RelVal RAW sample.

## Output via MessageLogger - configured, after much flailing, so that
## ONLY the LogVerbatim("CSCBadChannels") messages are sent to std:output.


import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCBadChannels = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

# --- MATCH GT TO RELEASE AND DATA SAMPLE

# This is OK for 72x real data
process.GlobalTag.globaltag = "GR_R_71_V1::All"

# --- NUMBER OF EVENTS ---  JUST ONE!

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

# --- MUST HAVE A DUMMY SOURCE

process.source = cms.Source("EmptySource",
 firstRun = cms.untracked.uint32(100001)
)

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring("ProductNotFound") )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.analyze = cms.EDAnalyzer("CSCRecoBadChannelsAnalyzer",
    readBadChannels = cms.bool(True),
    readBadChambers = cms.bool(False),
    CSCUseTimingCorrections = cms.bool(False),
    CSCUseGasGainCorrections = cms.bool(False)
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)
