import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.DQMStore = cms.Service("DQMStore")

process.tester = cms.EDAnalyzer("SMDQMClientExample")
process.pDQM = cms.Path(process.tester)

process.source = cms.Source("DQMHttpSource",
                            sourceURL = cms.untracked.string('http://cmsroc8.fnal.gov:50002/urn:xdaq-application:lid=29'),
                            DQMconsumerName = cms.untracked.string('Test Consumer'),
                            DQMconsumerPriority = cms.untracked.string('normal'),
                            headerRetryInterval = cms.untracked.int32(5),
                            maxDQMEventRequestRate = cms.untracked.double(1.0),
                            topLevelFolderName = cms.untracked.string('*')
                            )
