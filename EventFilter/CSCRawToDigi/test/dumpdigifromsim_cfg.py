# Dump digis from simulation
# --------------------------
# The simulation pathway is gen->sim->digi1->raw->digi2->[L1->HLT | reco]
# This config examines the digis at 'digi1' level using CSCDigiDump (in CSCDigitizer).

# cf. dumpdigisfromsimraw_cfg.py which examines 'digi2'-level digis.

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCSimDigiDump")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# You must supply an appropriate data set containing simulated digis
# (The 'relval' samples do tell you what they contain.)

# This is for 310p7 relval 20.05.2009

readFiles.extend([
 '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0004/EE15A7EC-E641-DE11-A279-001D09F29321.root',
 '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0003/AA3C9C4E-4441-DE11-BC27-001617C3B76A.root',
 '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0003/40989E55-4441-DE11-B896-000423D98868.root'
])
secFiles.extend([
])

process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")

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
process.p1 = cms.Path(process.cscSimDigiDump)

