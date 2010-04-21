import FWCore.ParameterSet.Config as cms

process = cms.Process('SPYDQM')

#source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
       'rfio:/castor/cern.ch/user/w/whyntie/data/spychannel/121834/edm/spydata_0001.root',
       #'rfio:/castor/cern.ch/user/w/whyntie/data/spychannel/121834/edm/spydata_0026.root',
       )
    )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
  )

# --- Message Logging ---
#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('SiStripSpyMonitor')
process.MessageLogger.suppressInfo = cms.untracked.vstring('SiStripSpyDigiConverter')
process.MessageLogger.suppressWarning = cms.untracked.vstring('SiStripSpyDigiConverter')
#process.MessageLogger.suppressDebug = cms.untracked.vstring('*')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_P_V8_34X::All'

process.load('DQM.SiStripMonitorHardware.SiStripSpyUnpacker_cfi')
process.load('DQM.SiStripMonitorHardware.SiStripSpyDigiConverter_cfi')
process.SiStripSpyUnpacker.InputProductLabel = cms.InputTag('source')
process.SiStripSpyUnpacker.StoreScopeRawDigis = cms.bool(True)

process.SiStripSpyDigiConverter.InputProductLabel = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyDigiConverter.StoreVirginRawDigis = cms.bool(True)
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
process.SiStripFEDEmulator.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','VirginRaw')
process.SiStripFEDEmulator.ByModule = cms.bool(True) #use the digis stored by module (i.e. detId)

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
#process.SiStripSpyMonitor.OutputErrors = "MinZero","MaxSat","LowRange","HighRange","LowDAC","HighDAC","OOS","OtherPbs","APVError","APVAddressError","NegPeds"
#process.SiStripSpyMonitor.WriteCabling = True

process.p = cms.Path(
    process.SiStripSpyUnpacker
    *process.SiStripSpyDigiConverter
    *process.SiStripFEDEmulator
    *process.SiStripSpyMonitor
    )

