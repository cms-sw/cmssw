# Unpack digis from simulated raw data and dump them
# --------------------------------------------------
# The simulation pathway is gen->sim->digi1->raw->digi2->[L1->HLT | reco]
# This config examines the digis at 'digi2' level using CSCDigiDump (in CSCDigitizer).

# cf. dumpdigisfromsim_cfg.py which examines 'digi1'-level digis.

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCSimRawToDigiDump")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V11::All'

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)
dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_IDEAL_V11_v1')

# You must supply an appropriate data set containing simulated 'raw' data.
# (The 'relval' samples do tell you what they contain.)

readFiles.extend([
   '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
   '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
   '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
   '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
])
secFiles.extend([
])

process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.dump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    empt = cms.InputTag(""),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
)

# Activate the following code to turn on LogDebug/LogTrace messages from CSCRawToDigi
##process.load("FWCore.MessageLogger.MessageLogger_cfi")
##process.MessageLogger.debugModules.append('muonCSCDigis')
##process.MessageLogger.categories.append('CSCRawToDigi')
##process.MessageLogger.cout = cms.untracked.PSet(
##  threshold     = cms.untracked.string('DEBUG'),
##  default       = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
##  FwkReport     = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
##  CSCRawToDigi  = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
##)

# Only dump 100 events
process.maxEvents = cms.untracked.PSet( input=cms.untracked.int32(100) )

process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonCSCDigis.Debug = True
process.muonCSCDigis.UseExaminer = True
process.p1 = cms.Path(process.muonCSCDigis+process.dump)
