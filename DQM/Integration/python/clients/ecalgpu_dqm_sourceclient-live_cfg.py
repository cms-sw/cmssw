import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("process", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

### Load cfis ###

if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

process.load("DQM.Integration.config.environment_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

process.load("FWCore.Modules.preScaler_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorTasks.ecalGpuTask_cfi")

### Individual module setups ###

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noTimeStamps = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalDQM = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring('EcalDQM'),
    destinations = cms.untracked.vstring('cerr',
        'cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.preScaler.prescaleFactor = 1

if not options.inputFiles:
    process.source.streamLabel = cms.untracked.string("streamDQMGPUvsCPU")

process.dqmEnv.subSystemFolder = 'Ecal'
process.dqmSaver.tag = 'EcalGPU'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'EcalGPU'
process.dqmSaverPB.runNumber = options.runNumber

process.ecalGpuTask.params.runGpuTask = True
process.ecalMonitorTask.workers = ['GpuTask']
process.ecalMonitorTask.workerParameters = cms.untracked.PSet(GpuTask = process.ecalGpuTask)
process.ecalMonitorTask.verbosity = 0
process.ecalMonitorTask.commonParameters.willConvertToEDM = False
process.ecalMonitorTask.commonParameters.onlineMode = True

# ecalMonitorTask always looks for EcalRawData collection when running, even when not in use
# Default value is cms.untracked.InputTag("ecalDigis")
# Tag is changed below to avoid multiple warnings per event
process.ecalMonitorTask.collectionTags.EcalRawData = cms.untracked.InputTag("hltEcalDigisLegacy")

# Streams used for online GPU validation
process.ecalMonitorTask.collectionTags.EBCpuDigi = cms.untracked.InputTag("hltEcalDigisLegacy", "ebDigis")
process.ecalMonitorTask.collectionTags.EECpuDigi = cms.untracked.InputTag("hltEcalDigisLegacy", "eeDigis")
process.ecalMonitorTask.collectionTags.EBGpuDigi = cms.untracked.InputTag("hltEcalDigisFromGPU", "ebDigis")
process.ecalMonitorTask.collectionTags.EEGpuDigi = cms.untracked.InputTag("hltEcalDigisFromGPU", "eeDigis")
process.ecalMonitorTask.collectionTags.EBCpuUncalibRecHit = cms.untracked.InputTag("hltEcalUncalibRecHitLegacy", "EcalUncalibRecHitsEB")
process.ecalMonitorTask.collectionTags.EECpuUncalibRecHit = cms.untracked.InputTag("hltEcalUncalibRecHitLegacy", "EcalUncalibRecHitsEE")
process.ecalMonitorTask.collectionTags.EBGpuUncalibRecHit = cms.untracked.InputTag("hltEcalUncalibRecHitFromSoA", "EcalUncalibRecHitsEB")
process.ecalMonitorTask.collectionTags.EEGpuUncalibRecHit = cms.untracked.InputTag("hltEcalUncalibRecHitFromSoA", "EcalUncalibRecHitsEE")
# RecHit GPU collection tags are temp placeholder
process.ecalMonitorTask.collectionTags.EBCpuRecHit = cms.untracked.InputTag("hltEcalRecHitWithoutTPs", "EcalRecHitsEB")
process.ecalMonitorTask.collectionTags.EECpuRecHit = cms.untracked.InputTag("hltEcalRecHitWithoutTPs", "EcalRecHitsEE")
process.ecalMonitorTask.collectionTags.EBGpuRecHit = cms.untracked.InputTag("hltEcalRecHitWithTPs", "EcalRecHitsEB")
process.ecalMonitorTask.collectionTags.EEGpuRecHit = cms.untracked.InputTag("hltEcalRecHitWithTPs", "EcalRecHitsEE")

### Paths ###

process.ecalMonitorPath = cms.Path(process.preScaler+process.ecalMonitorTask)
process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver + process.dqmSaverPB)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.dqmEndPath,process.dqmOutputPath)

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
print("Final Source settings:", process.source)
process = customise(process)
