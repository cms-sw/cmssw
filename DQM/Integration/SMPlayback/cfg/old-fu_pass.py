import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound','TooManyProducts','TooFewProducts'),
    makeTriggerResults = cms.untracked.bool(True)
    )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 180

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("DaqSource",
                    readerPluginName = cms.untracked.string('FUShmReader'),
                    evtsPerLS = cms.untracked.uint32(1000),
                    calculateFakeLumiSection = cms.untracked.bool(True)
                            )

process.pre1 = cms.EDFilter("Prescaler",
                          prescaleFactor = cms.int32(1),
                          prescaleOffset = cms.int32(0)
                          )

process.pathExtractor = cms.EDAnalyzer("HLTPathAnalyzer",
#                          TriggerResults = cms.InputTag("TriggerResults::HLT"),
                          TriggerResults = cms.InputTag("TriggerResults")
                          )



#process.playbackPath4DQM = cms.Path(process.pathExtractor * process.pre1)
process.playbackPath4DQM = cms.Path( process.pre1)

process.hltOutputDQM = cms.OutputModule("ShmStreamConsumer",
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
	 )
process.hltOutputHLTDQM = cms.OutputModule("ShmStreamConsumer",
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
            )

process.hltOutputExpress = cms.OutputModule("ShmStreamConsumer",
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
            )


process.e = cms.EndPath(process.hltOutputDQM*process.hltOutputHLTDQM*process.hltOutputExpress)
#process.f = cms.EndPath(process.hltOutputHLTDQM)

