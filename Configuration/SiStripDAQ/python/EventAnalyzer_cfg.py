# The following comments couldn't be translated into the new config version:

# ----- Services -----

import FWCore.ParameterSet.Config as cms

process = cms.Process("EventAnalyzer")
process.MLlog4cplus = cms.Service("MLlog4cplus")

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    log4cplus = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    suppressDebug = cms.untracked.vstring(),
    debugModules = cms.untracked.vstring('*'), ##@@ comment to suppress

    suppressInfo = cms.untracked.vstring()
)

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
    initialMessageBufferSize = cms.untracked.int32(1000000),
    compressionLevel = cms.int32(1),
    lumiSectionsPerUpdate = cms.double(1.0),
    useCompression = cms.bool(True)
)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    pset = cms.PSet(
        dummy = cms.untracked.int32(0)
    )
)

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

process.consumer = cms.EDFilter("ShmStreamConsumer",
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(7000000)
)

process.p1 = cms.Path(process.anal)
process.e1 = cms.EndPath(process.consumer)

