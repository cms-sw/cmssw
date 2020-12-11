## Process sim digi events with CSC rechit & segment builders - Tim Cox - 11.02.2015
## This version runs in 7_4_0_preX on a 7_3_0 simulated data DIGI relval sample.
##     -- USING OLD ALGO 'SK' --
## Run on  100  events of a 25ns PU TTbar sample

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
##process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
process.CSCIndexerESProducer.AlgoName = cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName = cms.string("CSCChannelMapperPostls1")

# --- MATCH GT TO RELEASE AND DATA SAMPLE

process.GlobalTag.globaltag = "MCRUN2_73_V5::All"

# --- NUMBER OF EVENTS

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring("ProductNotFound") )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## ttbar+pu is 200 events per file so need 5 for 1000 events
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            "/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/044157C8-A181-E411-AC04-002354EF3BD2.root"
 ,
"/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/0A963931-A181-E411-B4C5-0026189438DC.root",
"/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/145BE1DC-A181-E411-816D-0025905A609E.root",
"/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/14DDEDD0-A181-E411-9476-0026189438F8.root",
"/store/relval/CMSSW_7_3_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_73_V7_71XGENSIM-v1/00000/18F85F36-A181-E411-8AF4-0025905B85D6.root"
    )
)

# ME1/1A is  u n g a n g e d  postls1

process.CSCGeometryESModule.useGangedStripsInME1a = False
##process.CSCGeometryESModule.debugV = True
##process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

# Turn off some flags for CSCRecHitD that are turned ON in default config

process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
# Already defaults OFF...
## process.csc2DRecHits.CSCUseTimingCorrections = cms.bool(False)

# Switch input for CSCRecHitD to  s i m u l a t e d  digis

process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

# -- ACCESSING "DEEP" PARAMETERS OF THE ALGO IS TRICKY                                                                          
# THE FOLLOWING FOUND BY EXPLORING CONFIG WITH python -i                                                               

# switch to CSCSegAlgoSK
process.cscSegments.algo_type = cms.int32(1)         

# --- Activate LogVerbatim IN CSCSegment                                                                                         
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCSegment = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

# Path and EndPath def
process.reco = cms.Path(process.csc2DRecHits * process.cscSegments)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.reco, process.endjob)
