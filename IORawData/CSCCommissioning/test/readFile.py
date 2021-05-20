import FWCore.ParameterSet.Config as cms

process = cms.Process("reader")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("EmptySource",
      firstRun= cms.untracked.uint32(298313),
      numberEventsInLuminosityBlock = cms.untracked.uint32(200),
      numberEventsInRun       = cms.untracked.uint32(0)
)

process.rawDataCollector = cms.EDProducer('CSCFileReader',
      firstEvent  = cms.untracked.int32(0),
      FED856 = cms.untracked.vstring('RUI33'),
      RUI33 = cms.untracked.vstring('/tmp/barvic/csc_00298313_EmuRUI33_Local_000.raw')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string("/tmp/barvic/csc_00298313_GEM_Test.root"),
        outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path( process.rawDataCollector)

process.outpath = cms.EndPath(process.FEVT)
