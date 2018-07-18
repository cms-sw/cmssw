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
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0026.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0027.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0028.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0029.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0030.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0031.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0032.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0033.RUN00234874.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/234824/USC.00234824.0001.A.storageManager.00.0034.RUN00234874.root',
'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0000.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0001.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0002.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0003.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0004.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0005.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0006.RUN00298269.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/298270/USC.00298270.0001.A.storageManager.00.0007.RUN00298269.root',
        )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# --- Message Logging ---
#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')
#process.MessageLogger.debugModules = cms.untracked.vstring('')
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

# --- Define the path ---
process.p = cms.Path(
    process.SiStripSpyUnpacker
    *process.SiStripSpyDigiConverter
    )


# --- What to output ---
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyRawToDigis298270_TEST.root"),
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
