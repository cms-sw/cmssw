# from https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1O2OOperations
# to test the emulator from the .cms cluster retrieving the global tag
#
# The reference is L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# test.py starts here
# 
#
import FWCore.ParameterSet.Config as cms
process = cms.Process('L1')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(50))

readFiles = cms.untracked.vstring('file:Raw.root')
secFiles = cms.untracked.vstring() 
process.source = cms.Source('PoolSource',
                            fileNames=readFiles,
                            secondaryFileNames=secFiles
                            )

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.GlobalTag.globaltag = 'GR10_H_V4::All'

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']

process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    )
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
                                  fileName = cms.untracked.string('L1EmulatorFromRaw.root'),
                                  dataset = cms.untracked.PSet(dataTier = cms.untracked.string("\'GEN-SIM-DIGI-RAW-HLTDEBUG\'"),
                                                               filterName = cms.untracked.string('')
                                                               )
                                  )

process.out_step = cms.EndPath(process.output)
#
# test.py ends here
#

# Unpackers
process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")
process.load("EventFilter.DTTFRawToDigi.dttfunpacker_cfi")


# ------------------------------------------------------------------------------------------------
# IMPORTANT:
# ---------
#
# IF YOU WANT TO CONFIGURE THE EMULATOR VIA EventSetup (O2O mechanism or fake producer) the
# option initializeFromPSet in L1Trigger/CSCTrackFinder/python/csctfTrackDigis_cfi.py
# has to be set to False: initializeFromPSet = cms.bool(False)
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("csctfunpacker")
process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("dttfunpacker")
process.simCsctfTrackDigis.SectorProcessor.initializeFromPSet = cms.bool(False)

# if you want to read the DT stubs unpacked by CSCTF
# process.simCsctfTrackDigis.readDtDirect = cms.bool(True)
# process.simCsctfTrackDigis.mbProducer = cms.untracked.InputTag("csctfunpacker","DT")
# ------------------------------------------------------------------------------------------------

process.p = cms.Path( process.csctfunpacker
                     *process.dttfunpacker
                     *process.simCsctfTrackDigis)
