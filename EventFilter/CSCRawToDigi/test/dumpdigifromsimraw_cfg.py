# Unpack digis from simulated raw data and dump them
# --------------------------------------------------
# The simulation pathway is gen->sim->digi1->raw->digi2->[L1->HLT | reco]
# This config examines the digis at 'digi2' level using CSCDigiDump (in CSCDigitizer).

# cf. dumpdigisfromsim_cfg.py which examines 'digi1'-level digis.

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCSimRawToDigiDump")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V2::All'

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# You must supply an appropriate data set containing simulated 'raw' data.
# (The 'relval' samples do tell you what they contain.)

# This is for 310 relval samples 05.07.2009

readFiles.extend([
       '/store/relval/CMSSW_3_1_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/9AEF7C70-6066-DE11-B4C9-001D09F2516D.root',
       '/store/relval/CMSSW_3_1_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/74CB8381-6066-DE11-8C1F-001D09F241F0.root',
       '/store/relval/CMSSW_3_1_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/38770B61-DE66-DE11-BBF2-001D09F2545B.root'
])
secFiles.extend([
])

process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")


# Activate the following code to turn on LogDebug/LogTrace messages from CSCRawToDigi
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('muonCSCDigis')
process.MessageLogger.categories.append('CSCRawToDigi')
process.MessageLogger.categories.append('CSCCFEBData')
process.MessageLogger.cout = cms.untracked.PSet(
  threshold     = cms.untracked.string('DEBUG'),
  default       = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
  FwkReport     = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
##  CSCRawToDigi  = cms.untracked.PSet( limit = cms.untracked.int32(-1),
  CSCCFEBData  = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

# Only dump 10 events
process.maxEvents = cms.untracked.PSet( input=cms.untracked.int32(10) )

process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonCSCDigis.Debug = True
process.muonCSCDigis.UseExaminer = True
process.p1 = cms.Path(process.muonCSCDigis+process.cscDigiDump)
