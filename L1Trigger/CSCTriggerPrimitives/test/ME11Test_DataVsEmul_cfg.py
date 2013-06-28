# Configuration file to 
#  - read raw datafiles taken with ME11 test-stand local cosmic runs 
#  - unpack CSC digis
#  - run Trigger Primitives emulator
#  - compare A/C/LCTs stored in data vs. those found by the emulator using the CSCTriggerPrimitivesReader analyzer.

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCTPEmulator")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000000) )


# Hack to add the "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'], 'src/L1Trigger/CSCTriggerPrimitives/test'))

'''
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
         #'file:digi_test_7DCFEB.root'
         #'file:test_7DCFEB.root'
         #'file:test_clct_fifo_pretrig7.root'
         'file:test_2fix.root'
     )
)
'''

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('cscTriggerPrimitiveDigis','lctreader')


# pick up the geometry - ideal design geometry would suffice
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_38Y_V8::All'
#process.GlobalTag.globaltag = 'GR_R_61_V7::All'
process.GlobalTag.globaltag = 'DESIGN61_V11::All'



# CSC raw --> digi unpacker
# =========================
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.muonCSCDigis.InputObjects = "rawDataCollector"
# InputObjects = cms.InputTag("cscpacker","CSCRawData")
# for run 566 and 2008 data
# ErrorMask = cms.untracked.uint32(0xDFCFEFFF)



# CSC Trigger Primitives emulator
# ===============================
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
#process.cscTriggerPrimitiveDigis.clctParam07.clctHitPersist = 6
process.cscTriggerPrimitiveDigis.tmbParam.verbosity = 2
#process.cscTriggerPrimitiveDigis.tmbParam.tmbEarlyTbins = 0
#process.cscTriggerPrimitiveDigis.tmbParam.tmbL1aWindowSize = 9
process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
process.cscTriggerPrimitiveDigis.debugParameters = True
process.cscTriggerPrimitiveDigis.commonParam.gangedME1a = False


# CSC Trigger Primitives reader
# =============================
process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True
process.lctreader.CSCWireDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCWireDigi")
process.lctreader.CSCComparatorDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
process.lctreader.dataLctsIn = cms.bool(True) # if True, comparison of data & emulator would be performed
#process.lctreader.dataLctsIn = cms.bool(False) # if False - no comparison to data, histograms filled with emulator quantities


# Input source for raw datafiles from chamber testing
# =============================
process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent  = cms.untracked.int32(0),
        FED750 = cms.untracked.vstring('RUI01'),
        #RUI01 = cms.untracked.vstring('/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun6144s_000_130420_173808_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun6144s_001_130420_173808_UTC.raw')
        #RUI01 = cms.untracked.vstring('/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun13824s_000_130420_144345_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun13824s_001_130420_144345_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun13824s_002_130420_144345_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun13824s_003_130420_144345_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun13824s_004_130420_144345_UTC.raw')
        #RUI01 = cms.untracked.vstring('/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun6144s_000_130430_160359_UTC.raw')
        #RUI01 = cms.untracked.vstring('/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun4096s_000_130501_005900_UTC.raw')
        RUI01 = cms.untracked.vstring('/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun6144s_000_130501_121902_UTC.raw','/local/data/csc_00000001_EmuRUI01_ME11Test_ShortCosmicsRun6144s_001_130501_121902_UTC.raw')
  )
)


# Optional output
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("lct.root"),
    outputCommands = cms.untracked.vstring("keep *", 
        "drop *_DaqSource_*_*")
)


# ROOT file to save histograms
process.TFileService = cms.Service("TFileService",
    #fileName = cms.string('TPEHists_2fix.root')
    fileName = cms.string('TPEHists.root')
)


#process.cscdigiana = cms.EDAnalyzer('CSCDigiAnalizer',
#    fillTree = cms.untracked.bool(True)
#)


# Optional debuging tools:
#process.d=cms.EDAnalyzer('EventContentAnalyzer')
#process.Tracer = cms.Service("Tracer")


# Scheduler path
# ==============
process.p = cms.Path(process.muonCSCDigis * process.cscTriggerPrimitiveDigis * process.lctreader)
#process.p = cms.Path(process.muonCSCDigis * process.cscdigiana * process.cscTriggerPrimitiveDigis * process.lctreader)
#process.p = cms.Path(process.cscTriggerPrimitiveDigis * process.d * process.lctreader)
#process.ep = cms.EndPath(process.output)
