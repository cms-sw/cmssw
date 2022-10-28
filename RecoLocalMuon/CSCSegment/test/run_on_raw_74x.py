## Process  100  events in CSC segment builder - Tim Cox - 22.01.2015
## This version runs in  74X  on a Real RAW relval sample in 730 
## Now testing on FullSim+PU TTbar sample in 730
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
##       "/store/relval/CMSSW_7_3_0/SingleMu/RAW/GR_H_V43A_RelVal_zMu2012D-v1/00000/1ED9BE30-8481-E411-8AE9-002618943874.root"
         "/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/044157C8-A181-E411-AC04-002354EF3BD2.root"
    )
)

# -- ACCESSING 'DEEP' PARAMETERS OF THE ALGO IS TRICKY
# THE FOLLOWING FOUND BY EXPLORING CONFIG WITH python -i
# '3' is 4th algo CSCSegAlgoST; '0' and '1' are for ST_ME1234 and ST_ME1A configs
process.cscSegments.algo_psets[3].algo_psets[0].CSCDebug = cms.untracked.bool(True)
process.cscSegments.algo_psets[3].algo_psets[1].CSCDebug = cms.untracked.bool(True)


# --- Activate LogVerbatim IN CSCSegment
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCSegment = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)    

### --- ACTIVATE LogTrace IN VARIOUS MODULES - NEED TO COMPILE *EACH MODULE* WITH 
### scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
### LogTrace output goes to cout; all other output to "junk.log"

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.CSCRecHit=dict()
#process.MessageLogger.CSCSegment=dict()
#process.MessageLogger.CSCSegAlgoST=dict()

###  module label is something like "muonCSCDigis"...
#process.MessageLogger.debugModules = cms.untracked.vstring("*")
#process.MessageLogger.cerr.enable = False
#process.MessageLogger.files.junk = dict()
#process.MessageLogger.cout = cms.untracked.PSet(
#    enable    = cms.untracked.bool(True),
#    threshold = cms.untracked.string("DEBUG"),
#    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
#    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#    CSCRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#    CSCSegment = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
#  , CSCSegAlgoST = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
#)

# Path and EndPath def
process.unpack = cms.Path(process.muonCSCDigis)
process.reco = cms.Path(process.csc2DRecHits * process.cscSegments)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.unpack, process.reco, process.endjob)
