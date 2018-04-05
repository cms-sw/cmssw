import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process('SPYFEDEMULATOR')

##source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
#        'file:/eos/cms/store/user/jblee/SpyRawToDigis234824.root',
        'file:SpyMatchedEvents234824_TEST1.root'
#'file:/eos/cms/store/user/jblee/SpyRawToDigis234824_TEST.root'
        )
    )

## ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

## Global tag - see http://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
process.SiStripSpyDigiConverter.DiscardDigisWithWrongAPVAddress = True

## ---- FED Emulation ----
process.load('DQM.SiStripMonitorHardware.SiStripFEDEmulator_cfi')
process.SiStripFEDEmulator.SpyReorderedDigisTag = cms.InputTag('SiStripSpyEventMatcher','SpyReordered')
process.SiStripFEDEmulator.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyEventMatcher','SpyVirginRaw')
process.SiStripFEDEmulator.ByModule = cms.bool(True) #use the digis stored by module (i.e. detId)

#process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.p = cms.Path(
#     process.SiStripSpyUnpacker
#     *process.SiStripSpyDigiConverter
    #*process.profilerStart*
    process.SiStripFEDEmulator
    #*process.profilerStop 
    )

## --- What to output ---
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyMatched_FEDemulated234824_TEST.root"),
    outputCommands = cms.untracked.vstring(
       'keep *',
       #drop whatever collections from the above here - to save disk space!
       #'drop *_*_*_SPYEVENTMATCHING',
       #'drop *_SiStripSpyUnpacker_*_*',
       #'drop *_SiStripSpyDigiConverter_*_*',
       #'keep *_*_VirginRaw_*',
       #'drop *_TriggerResults_*_*',
       #'drop *_*_*_HLT'
       )
    )

process.e = cms.EndPath( process.output )
