import FWCore.ParameterSet.Config as cms

process = cms.Process('SPYFEDEMULATOR')

##source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        'rfio:/castor/cern.ch/user/w/whyntie/data/spychannel/121834/edm/spydata_0001.root',
        #'file:SpyRawToDigis.root'
        )
    )

## ---- Services ----
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

## Global tag - see http://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_P_V8_34X::All' # CMSSW 341

# --- The unpacking configuration ---
process.load('DQM.SiStripMonitorHardware.SiStripSpyUnpacker_cfi')
process.load('DQM.SiStripMonitorHardware.SiStripSpyDigiConverter_cfi')

## * Scope digi settings
process.SiStripSpyUnpacker.FEDIDs = cms.vuint32()                   #use a subset of FEDs or leave empty for all.
#process.SiStripSpy.FEDIDs = cms.vuint32(50, 187, 260, 356) #one from each partition
process.SiStripSpyUnpacker.InputProductLabel = cms.InputTag('source')
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
process.SiStripFEDEmulator.SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','Reordered')
process.SiStripFEDEmulator.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','VirginRaw')
process.SiStripFEDEmulator.ByModule = cms.bool(True) #use the digis stored by module (i.e. detId)

#process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.p = cms.Path(
    process.SiStripSpyUnpacker
    *process.SiStripSpyDigiConverter
    #*process.profilerStart*
    process.SiStripFEDEmulator
    #*process.profilerStop 
    )

## --- What to output ---
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyZeroSuppressed.root"),
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
