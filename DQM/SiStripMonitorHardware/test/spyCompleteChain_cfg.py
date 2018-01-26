#test configuration for the spy data unpacking code

import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process('SPYPROD')

# ---- Input data ----
# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/FEDSpyChannelData for more spy data.
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        # Spy data (raw) in edm format, as converted from .dat

'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0026.RUN00234874.root',
'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0027.RUN00234874.root',
'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0028.RUN00234874.root',
'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0029.RUN00234874.root',
'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0030.RUN00234874.root',
        )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# --- Message Logging ---
#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('')
#process.MessageLogger.suppressInfo = cms.untracked.vstring('')
#process.MessageLogger.suppressWarning = cms.untracked.vstring('')
#process.MessageLogger.suppressDebug = cms.untracked.vstring('')


# --- Conditions data ---
# Find the appropriate Global Tags at
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

# --- The unpacking configuration ---
process.load('DQM.SiStripMonitorHardware.SiStripSpyUnpacker_cfi')
process.load('DQM.SiStripMonitorHardware.SiStripSpyDigiConverter_cfi')

## * Scope digi settings
process.SiStripSpyUnpacker.FEDIDs = cms.vuint32()                   #use a subset of FEDs or leave empty for all.
#process.SiStripSpy.FEDIDs = cms.vuint32(50, 187, 260, 356) #one from each partition
process.SiStripSpyUnpacker.InputProductLabel = cms.InputTag('rawDataCollector')
process.SiStripSpyUnpacker.AllowIncompleteEvents = True
process.SiStripSpyUnpacker.StoreCounters = True
process.SiStripSpyUnpacker.StoreScopeRawDigis = cms.bool(True)      # Note - needs to be True for use in other modules.
## * Module digi settings
process.SiStripSpyDigiConverter.InputProductLabel = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyDigiConverter.StorePayloadDigis = True
process.SiStripSpyDigiConverter.StoreReorderedDigis = True
process.SiStripSpyDigiConverter.StoreModuleDigis = True
process.SiStripSpyDigiConverter.StoreAPVAddress = True
process.SiStripSpyDigiConverter.MinDigiRange = 100
process.SiStripSpyDigiConverter.MaxDigiRange = 1024
process.SiStripSpyDigiConverter.MinZeroLight = 0
process.SiStripSpyDigiConverter.MaxZeroLight = 1024
process.SiStripSpyDigiConverter.MinTickHeight = 0
process.SiStripSpyDigiConverter.MaxTickHeight = 1024
process.SiStripSpyDigiConverter.ExpectedPositionOfFirstHeaderBit = 6
process.SiStripSpyDigiConverter.DiscardDigisWithWrongAPVAddress = False


# ---- FED Emulation ----
process.load('DQM.SiStripMonitorHardware.SiStripFEDEmulator_cfi')
process.SiStripFEDEmulator.SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyReordered')
process.SiStripFEDEmulator.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyVirginRaw')
process.SiStripFEDEmulator.ByModule = cms.bool(True) #use the digis stored by module (i.e. detId)


## ---- SpyChannel display ----
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
# process.SiStripSpy.InputCompZeroSuppressedDigiLabel  = cms.InputTag("siStripDigis","ZeroSuppressed")

process.SiStripSpyDisplay.OutputFolderName = cms.string("Display")

# ---- DQM
process.DQMStore = cms.Service("DQMStore")

process.load('DQM.SiStripMonitorHardware.SiStripSpyMonitor_cfi')
process.SiStripSpyMonitor.SpyScopeRawDigisTag = cms.untracked.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyMonitor.SpyPedSubtrDigisTag = cms.untracked.InputTag('SiStripFEDEmulator','PedSubtrModuleDigis')
process.SiStripSpyMonitor.SpyAPVeTag = cms.untracked.InputTag('SiStripSpyDigiConverter','APVAddress')
process.SiStripSpyMonitor.FillWithLocalEventNumber = False
process.SiStripSpyMonitor.WriteDQMStore = True
process.SiStripSpyMonitor.DQMStoreFileName = "DQMStore.root"
#process.SiStripSpyMonitor.OutputErrors = "NoData","MinZero","MaxSat","LowRange","HighRange","LowDAC","HighDAC","OOS","OtherPbs","APVError","APVAddressError","NegPeds"
#process.SiStripSpyMonitor.OutputErrors = "MinZero","MaxSat","LowRange","HighRange","LowPb","HighPb","OOS","OtherPbs","APVError","APVAddressError","NegPeds"
#process.SiStripSpyMonitor.WriteCabling = True


## ------ TFileService
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('SpyDisplay.root')
    )


# --- Define the path ---
process.p = cms.Path(
    process.SiStripSpyUnpacker
    *process.SiStripSpyDigiConverter
    *process.SiStripFEDEmulator
#     *process.SiStripSpyMonitor
#     *process.SiStripSpyDisplay
    )


# --- What to output ---
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyRawToDigis234824_CH.root"),
    outputCommands = cms.untracked.vstring(
       'keep *',
       #'drop *',
       #'drop *_source_*_*',
       #'drop *_TriggerResults__SPYUNPACKTEST',
       #'drop *_*_ScopeRawDigis_*',
       #'drop *_*_Payload_*',
       #'drop *_*_Reordered_*',
       #'drop *_*_VirginRaw_*'
       #'drop *_*_TotalEventCount_*',
       #'drop *_*_L1ACount_*',
       #'drop *_*_APVAddress_*',
       )
    )

process.e = cms.EndPath( process.output )
