
#############################################
# Process declaration
#############################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripOnline")


#############################################
# Central services
#############################################

#process.MLlog4cplus = cms.Service("MLlog4cplus")
#process.MessageLogger = cms.Service("MessageLogger",
#    suppressWarning = cms.untracked.vstring(),
#    log4cplus = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    ),
#    suppressDebug = cms.untracked.vstring(),
#    debugModules = cms.untracked.vstring('*'), ##@@ comment to suppress
#    suppressInfo = cms.untracked.vstring()
#)

process.PrescaleService = cms.Service("PrescaleService",
    lvl1Labels = cms.vstring('DEFAULT'),
    prescaleTable = cms.VPSet()
)


#############################################
# DQM setup
#############################################

# DQM output via the shared memory
process.DQMStore = cms.Service("DQMStore")
process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
    initialMessageBufferSize = cms.untracked.int32(1000000),
    compressionLevel = cms.int32(1),
    lumiSectionInterval = cms.untracked.int32(2000000),
    lumiSectionsPerUpdate = cms.double(1.0),
    useCompression = cms.bool(True)
)


#############################################
# Tracker configuration database
#############################################

# config db parameters
process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    UsingDbCache = cms.untracked.bool(True),
    UsingDb = cms.untracked.bool(True),
    SharedMemory = cms.untracked.string('FEDSM00')
)

# use the config db configuration parameters
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")
process.NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")
process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)
process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")
# produce SiStripFecCabling and SiStripDetCabling out of SiStripFedCabling
process.sistripconn = cms.ESProducer("SiStripConnectivity")


#############################################
# Digis production
#############################################

# input module
process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    evtsPerLS = cms.untracked.uint32(50)
)

# tracker digi producer
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


#############################################
# modules & path for running without tracking
#############################################

# Commissioning source file production
process.histos = cms.EDFilter("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('digis'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    InputModuleLabel = cms.string('digis'),
    HistoUpdateFreq = cms.untracked.int32(10)
)

# the path to run for analysis without tracking
## process.p1 = cms.Path(
##     process.digis# *
## #    process.histos
## )


#############################################
# output
#############################################

## process.consumer = cms.EDFilter("ShmStreamConsumer",
##     outputCommands = cms.untracked.vstring(
##         'drop *' 
##         #'keep FEDRawDataCollection_*_*_*'
##     ),
##     SelectEvents = cms.untracked.PSet(
##         SelectEvents = cms.vstring('p1')
##     ),
##     compression_level = cms.untracked.int32(1),
##     use_compression = cms.untracked.bool(True),
##     max_event_size = cms.untracked.int32(25000000) ##@@ 440 FEDs @ 50kB each = 22 MB
## )

## process.e1 = cms.EndPath(
##     process.consumer
##     )

