## Run standalone CSCRecoBadChannelsAnalyzer - test bad strip channels - Tim Cox - 02.10.2014
## This version runs in 720pre6 on a real data RelVal RAW sample.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

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
