## Dump  100  events in CSC rechit builder - Tim Cox - 03.12.2012
## This version runs in 610preX on a real data RelVal RAW sample,
## and uses indexer and mapper algos.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# --- MATCH GT TO RELEASE AND DATA SAMPLE

process.GlobalTag.globaltag = 'GR_R_61_V1::All'

# --- NUMBER OF EVENTS --- 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_6_1_0_pre4-GR_P_V42_RelVal_mu2012C/SingleMu/RAW/v1/00000/F6AA1146-4F15-E211-8EE1-001A92810AD8.root'
    )
)

# Algorithm selection for Indexer & ChannelMapper

process.dummy1 = cms.ESSource("EmptyESSource",
                                  recordName = cms.string("CSCIndexerRecord"),
                                  firstValid = cms.vuint32(1),
                                  iovIsRunNotTime = cms.bool(True)
                              )

process.dummy2 = cms.ESSource("EmptyESSource",
                                  recordName = cms.string("CSCChannelMapperRecord"),
                                  firstValid = cms.vuint32(1),
                                  iovIsRunNotTime = cms.bool(True)
                              )

process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer", AlgoName = cms.string("CSCIndexerStartup") )
process.CSCChannelMapperESProducer = cms.ESProducer("CSCChannelMapperESProducer", AlgoName = cms.string("CSCChannelMapperStartup") )


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
    CSCRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCRecoConditions = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscValidation)

# Path and EndPath def
process.unpack = cms.Path(process.muonCSCDigis)
process.reco = cms.Path(process.csc2DRecHits)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.unpack, process.reco, process.endjob)

