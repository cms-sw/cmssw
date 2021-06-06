# to test the chain producer->emulator without using the global tag
import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Local RAW file
## process.source = cms.Source("DaqSource",
##     readerPluginName = cms.untracked.string('CSCFileReader'),
##     readerPset = cms.untracked.PSet(
##         firstEvent  = cms.untracked.int32(0),
##         tfDDUnumber = cms.untracked.int32(0),
##         FED760 = cms.untracked.vstring('RUI01'),
##         RUI01  = cms.untracked.vstring('/tmp/kkotov/rawfile')
##   )
## )


# GP's technique to test at P5 on cmsusr server: copy a file over and
# read it
readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource", fileNames = readFiles)
readFiles.extend((
        'file:Raw.root',
))

# read the streamer directly. Be careful: it WON'T work if your cmssw version
# if different from the one used to generate the file!
## readFiles = cms.untracked.vstring()
## process.source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
## ## data dat
## readFiles.extend( [
##     'file:/lookarea_SM/Data.00128960.0001.A.storageManager.00.0000.dat'
##  ] )



# CSC Track Finder emulator (copy-paste from L1Trigger/Configuration/python/SimL1Emulator_cff.py + little modifications)
# Little pieces of configuration, taken here and there
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")


# CSC TF (copy-paste L1Trigger/Configuration/python/L1RawToDigi_cff.py + little modifications)
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
process.csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
process.csctfDigis.producer = 'source'
#

import L1Trigger.CSCTrackFinder.csctfDigis_cfi

# ------------------------------------------------------------------------------------------------
# IMPORTANT:
# ---------
#
# IF YOU WANT TO CONFIGURE THE EMULATOR VIA EventSetup (O2O mechanism or fake producer) the
# option initializeFromPSet in L1Trigger/CSCTrackFinder/python/csctfTrackDigis_cfi.py
# has to be set to False: initializeFromPSet = cms.bool(False)
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("csctfDigis")
process.simCsctfTrackDigis.SectorProcessor.initializeFromPSet = cms.bool(False)
process.simCsctfTrackDigis.useDT = cms.bool(False)

# ------------------------------------------------------------------------------------------------
## # Following important parameters have to be set for singles by hand
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1a = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1b = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME2  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME3  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME4  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1a = cms.bool(False)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1d = cms.bool(False)
## process.simCsctfTrackDigis.SectorProcessor.singlesTrackPt     = cms.uint32(255)
## process.simCsctfTrackDigis.SectorProcessor.singlesTrackOutput = cms.uint32(1)

process.p = cms.Path(process.csctfDigis*process.simCsctfTrackDigis)

