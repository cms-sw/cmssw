import FWCore.ParameterSet.Config as cms

process = cms.Process("EVENTCONSUMER")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("EventStreamHttpReader",
                            sourceURL = cms.string('http://dvsrv-c2f37-01.cms:31110/urn:xdaq-application:lid=30'),
                            consumerName = cms.untracked.string('Test Consumer'),
                            headerRetryInterval = cms.untracked.int32(3),
                            maxEventRequestRate = cms.untracked.double(10.0),
                            SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*') ),
                            SelectHLTOutput = cms.untracked.string('hltOutputA'),
                            maxConnectTries = cms.untracked.int32(1)
                            )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout','log4cplus'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    log4cplus = cms.untracked.PSet(INFO = cms.untracked.PSet(reportEvery = cms.untracked.int32(250)),
                                                                   threshold = cms.untracked.string('INFO')
                                                                   )
                                    )

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path(process.contentAna)
