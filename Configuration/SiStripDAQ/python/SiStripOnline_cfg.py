# The following comments couldn't be translated into the new config version:

# ----- Services -----

import FWCore.ParameterSet.Config as cms

process = cms.Process("Commissioning")
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

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
    initialMessageBufferSize = cms.untracked.int32(1000000),
    compressionLevel = cms.int32(1),
    lumiSectionsPerUpdate = cms.double(1.0),
    useCompression = cms.bool(True)
)

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    MajorVersion = cms.untracked.uint32(0),
    ConfDb = cms.untracked.string(''),
    Partition = cms.untracked.string(''),
    UsingDb = cms.untracked.bool(True),
    MinorVersion = cms.untracked.uint32(0)
)

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    pset = cms.PSet(
        dummy = cms.untracked.int32(0)
    )
)

process.digis = cms.EDFilter("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('source'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(True),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(-1),
    ProductInstance = cms.untracked.string(''),
    CreateDigis = cms.untracked.bool(True)
)

process.histos = cms.EDFilter("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('digis'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    InputModuleLabel = cms.string('digis'),
    HistoUpdateFreq = cms.untracked.int32(10)
)

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

process.consumer = cms.EDFilter("ShmStreamConsumer",
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(7000000)
)

process.p1 = cms.Path(process.digis*process.histos)
process.e1 = cms.EndPath(process.consumer)

