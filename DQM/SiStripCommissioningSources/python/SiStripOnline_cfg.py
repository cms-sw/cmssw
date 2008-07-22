import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripOnline")
process.MLlog4cplus = cms.Service("MLlog4cplus")

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    log4cplus = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    suppressDebug = cms.untracked.vstring(),
    debugModules = cms.untracked.vstring('*'),
    suppressInfo = cms.untracked.vstring()
)

process.DQMStore = cms.Service("DQMStore")

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
    initialMessageBufferSize = cms.untracked.int32(1000000),
    compressionLevel = cms.int32(1),
    lumiSectionInterval = cms.untracked.int32(20),
    lumiSectionsPerUpdate = cms.double(1.0),
    useCompression = cms.bool(True)
)

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    UsingDbCache = cms.untracked.bool(True),
    UsingDb = cms.untracked.bool(True),
    SharedMemory = cms.untracked.string('')
)

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    evtsPerLS = cms.untracked.uint32(50)
)

process.digis = cms.EDProducer("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('source'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(True),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(-1),
    ProductInstance = cms.untracked.string(''),
    CreateDigis = cms.untracked.bool(True)
)

process.histos = cms.EDAnalyzer("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('digis'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    InputModuleLabel = cms.string('digis'),
    HistoUpdateFreq = cms.untracked.int32(10)
)

process.consumer = cms.OutputModule("ShmStreamConsumer",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep FEDRawDataCollection_*_*_*'),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(25000000)
)

process.p1 = cms.Path(process.digis*process.histos)
process.e1 = cms.EndPath(process.consumer)

