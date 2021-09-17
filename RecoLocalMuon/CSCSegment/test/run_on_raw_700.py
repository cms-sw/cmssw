## Dump  100  events in CSC segment builder - Tim Cox - 09.09.2013
## This version runs in 700pre3 on a real data RelVal RAW sample.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# --- MATCH GT TO RELEASE AND DATA SAMPLE

process.GlobalTag.globaltag = 'PRE_62_V8::All'

# --- NUMBER OF EVENTS --- 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_0_0_pre3/SingleMu/RAW/PRE_P62_V8_RelVal_mu2011B-v1/00000/127160CD-7215-E311-91F3-003048D15E14.root'
    )
)

# --- ACTIVATE LogTrace IN CSCRecHitD BUT NEED TO COMPILE IT WITH scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
# LogTrace output goes to cout; all other output to "junk.log"

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# module label is something like "muonCSCDigis"...
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.junk = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCSegAlgoST = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
)

#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscValidation)

# Path and EndPath def
process.unpack = cms.Path(process.muonCSCDigis)
process.reco = cms.Path(process.csc2DRecHits * process.cscSegments)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.unpack, process.reco, process.endjob)

