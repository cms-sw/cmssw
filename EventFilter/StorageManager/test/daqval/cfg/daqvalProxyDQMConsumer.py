import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.DQMStore = cms.Service("DQMStore")

process.tester = cms.EDAnalyzer("SMDQMClientExample")
process.pDQM = cms.Path(process.tester)

process.source = cms.Source("DQMHttpSource",
                            sourceURL = cms.untracked.string('http://dvsrv-c2f37-01.cms:31110/urn:xdaq-application:lid=30'),
                            DQMconsumerName = cms.untracked.string('Test Consumer'),
                            retryInterval = cms.untracked.int32(5),
                            topLevelFolderName = cms.untracked.string('*')
                            )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout','log4cplus'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    log4cplus = cms.untracked.PSet(INFO = cms.untracked.PSet(reportEvery = cms.untracked.int32(250)),
                                                                   threshold = cms.untracked.string('INFO')
                                                                   )
                                    )
