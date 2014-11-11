import FWCore.ParameterSet.Config as cms

process = cms.Process("ESDQM")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("FWCore.Modules.preScaler_cfi")

process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = cms.untracked.string('Ecal PreShower')

process.load("EventFilter.ESRawToDigi.esRawToDigi_cfi")
#process.ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
process.esRawToDigi.sourceTag = 'source'
process.esRawToDigi.debugMode = False

process.load('RecoLocalCalo/EcalRecProducers/ecalPreshowerRecHit_cfi')
process.ecalPreshowerRecHit.ESGain = cms.int32(2)
process.ecalPreshowerRecHit.ESBaseline = cms.int32(0)
process.ecalPreshowerRecHit.ESMIPADC = cms.double(50)
process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("esRawToDigi")
process.ecalPreshowerRecHit.ESRecoAlgo = cms.int32(0)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.preScaler.prescaleFactor = 1

#process.dqmInfoES = cms.EDAnalyzer("DQMEventInfo",
#                                   subSystemFolder = cms.untracked.string('EcalPreshower')
#                                   )

#process.load("DQMServices.Core.DQM_cfg")
#process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'EcalPreshower'
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/es_reference.root'

process.load("DQM/EcalPreshowerMonitorModule/EcalPreshowerMonitorTasks_cfi")
process.ecalPreshowerIntegrityTask.ESDCCCollections = cms.InputTag("esRawToDigi")
process.ecalPreshowerIntegrityTask.ESKChipCollections = cms.InputTag("esRawToDigi")
process.ecalPreshowerIntegrityTask.ESDCCCollections = cms.InputTag("esRawToDigi")
process.ecalPreshowerIntegrityTask.ESKChipCollections = cms.InputTag("esRawToDigi")
process.ecalPreshowerOccupancyTask.DigiLabel = cms.InputTag("esRawToDigi")
process.ecalPreshowerPedestalTask.DigiLabel = cms.InputTag("esRawToDigi")
process.ecalPreshowerRawDataTask.ESDCCCollections = cms.InputTag("esRawToDigi")
process.ecalPreshowerTimingTask.DigiLabel = cms.InputTag("esRawToDigi")
process.ecalPreshowerTrendTask.ESDCCCollections = cms.InputTag("esRawToDigi")

process.load("DQM/EcalPreshowerMonitorClient/EcalPreshowerMonitorClient_cfi")
del process.dqmInfoES
process.p = cms.Path(process.preScaler*
               process.esRawToDigi*
               process.ecalPreshowerRecHit*
               process.ecalPreshowerDefaultTasksSequence*
               process.dqmEnv*
               process.ecalPreshowerMonitorClient*
               process.dqmSaver)

