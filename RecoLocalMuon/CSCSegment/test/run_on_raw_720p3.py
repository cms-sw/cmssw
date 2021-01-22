## Dump  1000  events in CSC segment builder - Tim Cox - 22.08.2014
## This version runs in 720pre3 on a real data RelVal RAW sample.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# --- MATCH GT TO RELEASE AND DATA SAMPLE

# This is OK for 72x real data
process.GlobalTag.globaltag = 'GR_R_71_V1::All'

# --- NUMBER OF EVENTS --- 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RAW/GR_H_V37_RelVal_mu2012D-v1/00000/0011FE1A-0E19-E411-BE99-0025905A607E.root'
    )
)

# --- ACTIVATE LogTrace IN VARIOUS MODULES - NEED TO COMPILE *EACH MODULE* WITH 
# scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
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
    CSCSegment = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
  , CSCSegAlgoST = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscValidation)

# Path and EndPath def
process.unpack = cms.Path(process.muonCSCDigis)
process.reco = cms.Path(process.csc2DRecHits * process.cscSegments)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.unpack, process.reco, process.endjob)

