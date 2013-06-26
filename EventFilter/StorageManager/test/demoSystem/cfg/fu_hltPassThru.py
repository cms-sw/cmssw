import FWCore.ParameterSet.Config as cms

process = cms.Process("FU")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True),
                                     makeTriggerResults = cms.untracked.bool(True),
                                     Rethrow = cms.untracked.vstring('ProductNotFound',
                                                                     'TooManyProducts',
                                                                     'TooFewProducts')
                                     )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout','log4cplus'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
                                    log4cplus = cms.untracked.PSet(INFO = cms.untracked.PSet(reportEvery = cms.untracked.int32(250)),
                                                                   threshold = cms.untracked.string('INFO')
                                                                   )
                                    )

process.source = cms.Source("DaqSource",
                            readerPluginName = cms.untracked.string('FUShmReader'),
                            evtsPerLS = cms.untracked.uint32(10000000)
                            )

process.pre1 = cms.EDFilter("Prescaler",
                            prescaleFactor = cms.int32(1),
                            prescaleOffset = cms.int32(0)
                            )

process.playbackPath4DQM = cms.Path(process.pre1)

process.hltOutputDQM = cms.OutputModule("ShmStreamConsumer",
                                        use_compression = cms.untracked.bool(True),
                                        SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*') )
                                        )

process.e = cms.EndPath(process.hltOutputDQM)

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
                                            initialMessageBufferSize = cms.untracked.int32(1000000),
                                            lumiSectionsPerUpdate = cms.double(1.0),
                                            useCompression = cms.bool(True),
                                            compressionLevel = cms.int32(1)
                                            )
