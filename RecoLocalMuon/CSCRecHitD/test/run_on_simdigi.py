## Dump  10  events in CSC rechit builder - Tim Cox - 07.11.2012
## This version runs in 6_0_1_PostLS1 on a simulated data DIGI sample.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
##process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

# --- MATCH GT TO RELEASE AND DATA SAMPLE

process.GlobalTag.globaltag = "POSTLS161_V11::All"

# --- NUMBER OF EVENTS

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.options   = cms.untracked.PSet( SkipEvent = cms.untracked.vstring("ProductNotFound") )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.source    = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
         "/store/relval/CMSSW_6_0_1_PostLS1v1-PU_POSTLS161_V10/RelValSingleMuPt100_UPGpostls1/GEN-SIM-DIGI-RAW/v1/00000/38F76FF5-0126-E211-BAA2-002618943971.root"
    )
)

# ME1/1A is  u n g a n g e d  Post-LS1

process.CSCGeometryESModule.useGangedStripsInME1a = False
##process.CSCGeometryESModule.debugV = True
##process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

# Turn off some flags for CSCRecHitD that are turned ON in default config

process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseTimingCorrections = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

# Switch input for CSCRecHitD to  s i m u l a t e d  digis

process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")


# --- TO ACTIVATE LogTrace IN CSCRecHitD NEED TO COMPILE IT WITH scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
# LogTrace output goes to cout; all other output to "junk.log"

process.load("FWCore.MessageLogger.MessageLogger_cfi")
##process.MessageLogger.categories.append("CSCGeometry")
process.MessageLogger.categories.append("CSCRecHit")
##process.MessageLogger.categories.append("CSCRecHitDBuilder")
##process.MessageLogger.categories.append("CSCMake2DRecHit")
## process.MessageLogger.categories.append("CSCHitFromStripOnly")
## process.MessageLogger.categories.append("CSCRecoConditions")
# module label is something like "muonCSCDigis"...
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
    , CSCRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCRecHitDBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCMake2DRecHit = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCHitFromStripOnly = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##    , CSCRecoConditions = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)



# Path and EndPath def
process.reco = cms.Path(process.csc2DRecHits)
process.endjob = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.reco, process.endjob)
