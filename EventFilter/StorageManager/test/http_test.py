import FWCore.ParameterSet.Config as cms

process = cms.Process("EVENTCONSUMER")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("EventStreamHttpReader",
                            sourceURL = cms.string('http://cmsroc8.fnal.gov:50002/urn:xdaq-application:lid=29'),
                            consumerName = cms.untracked.string('Test Consumer'),
                            consumerPriority = cms.untracked.string('normal'),
                            headerRetryInterval = cms.untracked.int32(3),
                            maxEventRequestRate = cms.untracked.double(10.0),
                            SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*DQM') ),
                            SelectHLTOutput = cms.untracked.string('out4DQM'),
                            maxConnectTries = cms.untracked.int32(1)
                            )

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path(process.contentAna)
