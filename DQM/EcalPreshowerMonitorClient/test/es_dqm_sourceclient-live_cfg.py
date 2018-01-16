import FWCore.ParameterSet.Config as cms

process = cms.Process("ESDQM")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("FWCore.Modules.preScaler_cfi")

process.load("DQM.Integration.test.inputsource_cfi")

process.load('EventFilter/ESRawToDigi/esRawToDigi_cfi')
process.esRawToDigi.sourceTag = 'source'
process.esRawToDigi.debugMode = False

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.preScaler.prescaleFactor = 1

process.dqmInfoES = DQMStep1Module('DQMEventInfo',
                                   subSystemFolder = cms.untracked.string('EcalPreshower')
                                   )

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'EcalPreshower'

process.load("DQM/EcalPreshowerMonitorModule/EcalPreshowerMonitorTasks_cfi")
process.load("DQM/EcalPreshowerMonitorClient/EcalPreshowerMonitorClient_cfi")

process.p = cms.Path(process.preScaler*process.esRawToDigi*process.ecalPreshowerDefaultTasksSequence*process.ecalPreshowerMonitorClient*process.dqmEnv*process.dqmSaver*process.dqmInfoES)

