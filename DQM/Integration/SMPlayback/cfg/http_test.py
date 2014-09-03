import FWCore.ParameterSet.Config as cms

process = cms.Process("EVENTCONSUMER")

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("EventStreamHttpReader", 
               sourceURL = cms.string('http://srv-C2D05-12:50082/urn:xdaq-application:lid=29'),
               max_event_size=cms.untracked.int32(7000000),
               max_queue_depth=cms.untracked.int32(5),
               consumerName = cms.untracked.string('Test Consumer'),
               consumerPriority = cms.untracked.string('normal'),
               headerRetryInterval = cms.untracked.int32(3),
               maxEventRequestRate = cms.untracked.double(2.5),
               SelectHLTOutput = cms.untracked.string('hltOutputDQM'),
               SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*') ),
               maxConnectTries = cms.untracked.int32(1) 
              )

process.mon1 = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.mon1)

