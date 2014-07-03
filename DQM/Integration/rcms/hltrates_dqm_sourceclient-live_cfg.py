import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQMResults')
process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(1000.0)
#process.EventStreamHttpReader.sourceURL = cms.string('http://srv-c2c07-13.cms:11100/urn:xdaq-application:lid=50')
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/hlt_reference.root"

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.test.environment_cfi")
process.dqmSaver.version = 2
#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" ) # for muon hlt dqm
#SiStrip Local Reco
#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
#process.TkDetMap = cms.Service("TkDetMap")

#---- for P5 (online) DB access
#process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.load("DQM.HLTEvF.TrigResRateMon_cfi")

#process.p = cms.EndPath(process.hlts+process.hltsClient)
process.p = cms.EndPath(process.trRateMon)


process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
process.EventStreamHttpReader.consumerName = 'HLTTrigerResults DQM Consumer'
process.dqmEnv.subSystemFolder = 'HLT/TrigResults'
#process.hltResults.plotAll = True

