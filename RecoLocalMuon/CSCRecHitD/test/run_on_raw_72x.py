## Process  100  events in CSC rechit builder - Tim Cox - 06.10.2014
## This version runs in 720pre6, 7, 73X IBs  on a real data RelVal RAW sample.
## Change Geometry_cff to GeometryDB_cff and update GT July.2022

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

# --- MATCH GT TO RELEASE AND DATA SAMPLE
# 2022
process.GlobalTag.globaltag = 'auto:phase1_2022_realistic'

# --- NUMBER OF EVENTS --- 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring("ProductNotFound") )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_7_2_0_pre6/SingleMu/RAW/GR_H_V38A_RelVal_mu2012D-v1/00000/001E161E-6F3F-E411-BD55-0025905A60D0.root" )
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
    CSCRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
)

# Path and EndPath def
process.unpack = cms.Path(process.muonCSCDigis)
process.reco = cms.Path(process.csc2DRecHits)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.unpack, process.reco, process.endjob)
