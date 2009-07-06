# Unpack digis from raw data and dump them
# --------------------------------------------------
# This config examines the digis using CSCDigiDump (in CSCDigitizer).

# Global/CRUZET data 03.07.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCRawToDigiDump")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_31X_V2P::All'

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

readFiles.extend( [
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/CAC42549-7E67-DE11-BBE7-000423D6CA42.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/8A408B6B-7467-DE11-89BF-001D09F253FC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/50F50574-7B67-DE11-9064-000423D99394.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/24A36DB8-7A67-DE11-9FB5-000423D99658.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/04A9F3ED-8367-DE11-9A5E-001D09F295FB.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/02B7336E-7B67-DE11-8B46-000423D98A44.root',
       '/store/data/Commissioning09/Cosmics/RECO/v4/000/102/173/00D506D8-7C67-DE11-931A-000423D60FF6.root' ] );


secFiles.extend( [
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/F04B7D65-4E67-DE11-AFE3-000423D952C0.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/ECC3F663-4E67-DE11-ABD0-000423D94A20.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/D650D167-4E67-DE11-853D-000423D98A44.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/D29CE383-4B67-DE11-9FF6-000423D951D4.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/CCC5CAD4-4A67-DE11-B9CB-000423D99660.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/8C905A66-4E67-DE11-9A59-000423D6A6F4.root',
       '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/173/20221969-4E67-DE11-915E-000423D99BF2.root'] );


process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.dump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    empt = cms.InputTag(""),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
)

# Activate the following code to turn on LogDebug/LogTrace messages from CSCRawToDigi
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('muonCSCDigis')
##process.MessageLogger.categories.append('CSCRawToDigi')
process.MessageLogger.categories.append('CSCCFEBData')
process.MessageLogger.cout = cms.untracked.PSet(
  threshold     = cms.untracked.string('DEBUG'),
  default       = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
  FwkReport     = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
##  CSCRawToDigi  = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
  CSCCFEBData   = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

# Only dump 10 events
process.maxEvents = cms.untracked.PSet( input=cms.untracked.int32(10) )

process.muonCSCDigis.Debug = cms.untracked.bool(True)

process.p1 = cms.Path(process.muonCSCDigis+process.dump)
