import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process('SPYDQM')

#source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
       'file:/eos/cms/store/user/jblee/SpyFEDemulated234824.root',
       )
    )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
  )

# --- Message Logging ---
#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('SiStripSpyMonitor')
process.MessageLogger.suppressInfo = cms.untracked.vstring('SiStripSpyDigiConverter')
process.MessageLogger.suppressWarning = cms.untracked.vstring('SiStripSpyDigiConverter')
#process.MessageLogger.suppressDebug = cms.untracked.vstring('*')


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")


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
process.SiStripFEDEmulator.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyVirginRaw')
process.SiStripFEDEmulator.ByModule = cms.bool(True) #use the digis stored by module (i.e. detId)


process.load('DQM.SiStripMonitorHardware.SiStripSpyEventSummaryProducer_cfi')
process.SiStripSpyEventSummary.RawDataTag = cms.InputTag('rawDataCollector')
process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")
process.CommissioningHistos.CommissioningTask = 'PEDESTALS'  # <-- run type taken from even
process.CommissioningHistos.InputModuleLabel = 'SiStripSpyDigiConverter'  # output label fr
process.CommissioningHistos.SummaryInputModuleLabel = 'SiStripSpyEventSummary'

# ---- DQM
process.DQMStore = cms.Service("DQMStore")

process.load('DQM.SiStripMonitorHardware.SiStripSpyMonitor_cfi')
process.SiStripSpyMonitor.SpyScopeRawDigisTag = cms.untracked.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyMonitor.SpyPedSubtrDigisTag = cms.untracked.InputTag('SiStripFEDEmulator','PedSubtrModuleDigis')
process.SiStripSpyMonitor.SpyAPVeTag = cms.untracked.InputTag('SiStripSpyDigiConverter','APVAddress')
process.SiStripSpyMonitor.FillWithLocalEventNumber = True
process.SiStripSpyMonitor.WriteDQMStore = True
process.SiStripSpyMonitor.DQMStoreFileName = "DQMStore.root"
# process.SiStripSpyMonitor.OutputErrors = "NoData","MinZero","MaxSat","LowRange","HighRange","LowDAC","HighDAC","OOS","OtherPbs","APVError","APVAddressError","NegPeds"
# process.SiStripSpyMonitor.OutputErrors = "MinZero","MaxSat","LowRange","HighRange","LowDAC","HighDAC","OOS","OtherPbs","APVError","APVAddressError","NegPeds"
# process.SiStripSpyMonitor.WriteCabling = True

process.p = cms.Path(
#     process.SiStripSpyUnpacker
#     *process.SiStripSpyDigiConverter
#    process.SiStripFEDEmulator
    process.SiStripSpyMonitor
#    *process.SiStripSpyEventSummary
#     *process.CommissioningHistos
    )

