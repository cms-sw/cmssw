import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'EventContent DQM Consumer'
process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(0.1)
#process.EventStreamHttpReader.sourceURL = cms.string('http://dqm-c2d07-30:22100/urn:xdaq-application:lid=30')
#process.EventStreamHttpReader.sourceURL = cms.string('http://dqm-c2d07-30.cms:50082/urn:xdaq-application:lid=29')

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

#### Sub-system configuration follows
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.evfDQMmodulesPath = cms.Path(
			      process.dump
)
process.schedule = cms.Schedule(process.evfDQMmodulesPath)
