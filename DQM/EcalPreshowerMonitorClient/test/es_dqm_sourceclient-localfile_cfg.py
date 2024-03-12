import FWCore.ParameterSet.Config as cms

process = cms.Process("ESDQM")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("FWCore.Modules.preScaler_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("NewEventStreamFileReader",
                            fileNames = cms.untracked.vstring(
    'file:/esdata/esdev-data/es_cabling/sm/00/closed/es_cabling.00118701.0001.A.storageManager.00.0000.dat'
    )
                            )

#process.load('EventFilter/ESRawToDigi/esRawToDigi_cfi')
#process.esRawToDigi.sourceTag = 'source'
#process.esRawToDigi.debugMode = False
#process.esRawToDigi.debugMode = True

import EventFilter.ESRawToDigi.esRawToDigi_cfi
process.ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
process.ecalPreshowerDigis.sourceTag = 'source'
process.ecalPreshowerDigis.debugMode = False

process.load('RecoLocalCalo/EcalRecProducers/ecalPreshowerRecHit_cfi')
process.ecalPreshowerRecHit.ESGain = cms.int32(2)
process.ecalPreshowerRecHit.ESBaseline = cms.int32(0)
process.ecalPreshowerRecHit.ESMIPADC = cms.double(50)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.preScaler.prescaleFactor = 1

process.load("DQMServices.Core.DQM_cfg")
process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
                                  saveByRun = cms.untracked.int32(1),
                                  dirName = cms.untracked.string('.'),
                                  saveAtJobEnd = cms.untracked.bool(True),
                                  convention = cms.untracked.string('Online'),
                                  referenceHandling = cms.untracked.string('all')
                                  )

process.load("DQM/EcalPreshowerMonitorModule/EcalPreshowerMonitorTasks_cfi")
process.load("DQM/EcalPreshowerMonitorClient/EcalPreshowerLocalMonitorClient_cfi")

process.p = cms.Path(process.preScaler*process.ecalPreshowerDigis*process.ecalPreshowerRecHit*process.ecalPreshowerLocalTasksSequence*process.ecalPreshowerLocalMonitorClient*process.dqmSaver*process.dqmInfoES)

process.DQM.collectorHost = 'srv-c2c04-07'
process.DQM.collectorPort = 9190
process.DQM.debug = True
#process.DQMStore.verbose = 1
# foo bar baz
# F4Q2DLcglAMNg
# Tio6HHxMl6Lyd
