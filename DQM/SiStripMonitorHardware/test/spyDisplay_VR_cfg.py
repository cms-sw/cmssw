## Configuration for testing the FED processing monitor module
##=============================================================

import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process('SPYDISPLAY')

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
#       'file:/afs/cern.ch/work/j/jblee/public/SpyChannel/CMSSW_9_4_0/src/DQM/SiStripMonitorHardware/test/SpyMatchedEvents298270.root',
       'file:SpyMatchedEvents298270_TEST.root'
       )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

## ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")

## ---- Conditions ----

## Global tag see SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

## ---- Retrieve the ZS digis from the mainline ----
## ---- if running on matched events     ----
process.load('EventFilter.SiStripRawToDigi.SiStripDigis_cfi')
#process.siStripDigis.ProductLabel = cms.InputTag('source')
#process.siStripDigis.UnpackCommonModeValues = cms.bool(True)


## ---- SpyChannel Monitoring ----
## For my plugin for the spy channel monitoring
process.load('DQM.SiStripMonitorHardware.SiStripSpyDisplay_cfi')

## Select the detIDs of choice here
process.SiStripSpyDisplay.detIDs = cms.vuint32(
470079220, 470083621, 470083622, 470083625, 470083626, 470083653, 470083654, 470083657, 470083658, 470083684, 470083688, 470083692, 470083716, 470083720, 470083724, 470083728, 470083780, 470083784, 470083788,
369124565, 369124566, 369124569, 369124570, 369124573, 369124574, 369125557, 369125558, 369125561, 369125562, 369125565, 369125566, 369125573, 369125574, 369125577, 369125578, 369125581, 369125582, 369125589,
402672433, 402672434, 402672265, 402672266, 402672269, 402672270, 402672273, 402672274, 402672261, 402672262, 402672301, 402672302, 402672305, 402672306, 402672901, 402672902, 402672905, 402672906, 402672945,
436298448, 436298452, 436298456, 436298404, 436298408, 436298412, 436298416, 436298420, 436298424, 436298372, 436298376, 436298380, 436298384, 436298388, 436298392, 436298916, 436298920, 436298924, 436298928,
    )
process.SiStripSpyDisplay.InputScopeModeRawDigiLabel = cms.InputTag("SiStripSpyEventMatcher","SpyScope")
process.SiStripSpyDisplay.InputPayloadRawDigiLabel   = cms.InputTag("SiStripSpyEventMatcher", "SpyPayload")
process.SiStripSpyDisplay.InputReorderedPayloadRawDigiLabel = cms.InputTag("SiStripSpyEventMatcher", "SpyReordered")
process.SiStripSpyDisplay.InputReorderedModuleRawDigiLabel = cms.InputTag("SiStripSpyEventMatcher", "SpyVirginRaw")
#process.SiStripSpyDisplay.InputPedestalsLabel               = cms.InputTag("SiStripFEDEmulator","ModulePedestals")
#process.SiStripSpyDisplay.InputNoisesLabel                  = cms.InputTag("SiStripFEDEmulator","ModuleNoises")
#process.SiStripSpyDisplay.InputPostPedestalRawDigiLabel     = cms.InputTag("SiStripFEDEmulator","PedSubtrModuleDigis")
#process.SiStripSpyDisplay.InputPostCMRawDigiLabel           = cms.InputTag("SiStripFEDEmulator","CMSubtrModuleDigis")
#process.SiStripSpyDisplay.InputZeroSuppressedDigiLabel      = cms.InputTag("SiStripFEDEmulator","ZSModuleDigis")
##mainline data - if running on matched events
# process.SiStripSpy.InputCompZeroSuppressedDigiLabel  = cms.InputTag("siStripDigis","ZeroSuppressed")
process.SiStripSpyDisplay.InputCompVirginRawDigiLabel  = cms.InputTag("siStripDigis","VirginRaw")
process.SiStripSpyDisplay.OutputFolderName = cms.string("Display")

## ---- Sequence ----
process.p = cms.Path(
    process.siStripDigis*    ## if running on matched events
    process.SiStripSpyDisplay
    )

## ------ TFileService
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('SpyDisplay298270_Test.root')
    )

