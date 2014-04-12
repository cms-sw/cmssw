## Configuration for testing the FED processing monitor module
##=============================================================

import FWCore.ParameterSet.Config as cms

process = cms.Process('SPYDISPLAY')

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
       'file:SpyZeroSuppressed.root',
       )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

## ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")

## ---- Conditions ----
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## Global tag see SWGuideFrontierConditions
process.GlobalTag.globaltag = 'GR09_P_V8_34X::All'  ## For CMSSW 34X

## ---- Retrieve the ZS digis from the mainline ----
## ---- if running on matched events     ----
#process.load('EventFilter.SiStripRawToDigi.SiStripDigis_cfi')
#process.siStripDigis.ProductLabel = cms.InputTag('source')
#process.siStripDigis.UnpackCommonModeValues = cms.bool(True)


## ---- SpyChannel Monitoring ----
## For my plugin for the spy channel monitoring
process.load('DQM.SiStripMonitorHardware.SiStripSpyDisplay_cfi')

## Select the detIDs of choice here
process.SiStripSpyDisplay.detIDs = cms.vuint32(
    470079220
    )
process.SiStripSpyDisplay.InputScopeModeRawDigiLabel = cms.InputTag("SiStripSpyUnpacker","ScopeRawDigis")
process.SiStripSpyDisplay.InputPayloadRawDigiLabel   = cms.InputTag("SiStripSpyDigiConverter", "Payload")
process.SiStripSpyDisplay.InputReorderedPayloadRawDigiLabel = cms.InputTag("SiStripSpyDigiConverter", "Reordered")
process.SiStripSpyDisplay.InputReorderedModuleRawDigiLabel = cms.InputTag("SiStripSpyDigiConverter", "VirginRaw")
process.SiStripSpyDisplay.InputPedestalsLabel               = cms.InputTag("SiStripFEDEmulator","ModulePedestals")
process.SiStripSpyDisplay.InputNoisesLabel                  = cms.InputTag("SiStripFEDEmulator","ModuleNoises")
process.SiStripSpyDisplay.InputPostPedestalRawDigiLabel     = cms.InputTag("SiStripFEDEmulator","PedSubtrModuleDigis")
process.SiStripSpyDisplay.InputPostCMRawDigiLabel           = cms.InputTag("SiStripFEDEmulator","CMSubtrModuleDigis")
process.SiStripSpyDisplay.InputZeroSuppressedDigiLabel      = cms.InputTag("SiStripFEDEmulator","ZSModuleDigis")
##mainline data - if running on matched events
#process.SiStripSpy.InputCompZeroSuppressedDigiLabel  = cms.InputTag("siStripDigis","ZeroSuppressed")

process.SiStripSpyDisplay.OutputFolderName = cms.string("Display")

## ---- Sequence ----
process.p = cms.Path(
    #process.siStripDigis*    ## if running on matched events
    process.SiStripSpyDisplay
    )

## ------ TFileService
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('SpyDisplay.root')
    )

