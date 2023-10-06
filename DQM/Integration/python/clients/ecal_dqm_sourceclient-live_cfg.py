### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
import FWCore.ParameterSet.Config as cms
from EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi import * # To monitor LHC status, e.g. to mask trigger primitives quality alarm during Cosmics
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
    from DQM.Integration.config.inputsource_cfi import options, set_BeamSplashRun_settings

process.load("DQM.Integration.config.environment_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("FWCore.Modules.preScaler_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
#process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("Geometry.EcalMapping.EcalMapping_cfi")
#process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
#process.load("L1Trigger.Configuration.L1RawToDigi_cff")
#process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")

process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi")
#process.load("RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi")
#process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")
#process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
#process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
#process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")
#process.load("RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff")
#process.load("RecoEcal.EgammaClusterProducers.reducedRecHitsSequence_cff")

process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorClient.EcalMonitorClient_cfi")

### Individual module setups ###

# Use the ratio timing method for the online DQM
process.ecalMultiFitUncalibRecHit.cpu.algoPSet.timealgo = cms.string("RatioMethod")
process.ecalMultiFitUncalibRecHit.cpu.algoPSet.outOfTimeThresholdGain12pEB = cms.double(5.)
process.ecalMultiFitUncalibRecHit.cpu.algoPSet.outOfTimeThresholdGain12mEB = cms.double(5.)
process.ecalMultiFitUncalibRecHit.cpu.algoPSet.outOfTimeThresholdGain61pEB = cms.double(5.)
process.ecalMultiFitUncalibRecHit.cpu.algoPSet.outOfTimeThresholdGain61mEB = cms.double(5.)

process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler",
    cosmics = cms.untracked.uint32(1),
    physics = cms.untracked.uint32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalLaserDbService = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
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
    categories = cms.untracked.vstring('EcalDQM', 
        'EcalLaserDbService'),
    destinations = cms.untracked.vstring('cerr', 
        'cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
if not unitTest:
  if options.BeamSplashRun:
      set_BeamSplashRun_settings( process.source )
      process.ecalMonitorTask.workerParameters.TimingTask.params.splashSwitch = True
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeMap.zaxis.high = cms.untracked.double(30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeMap.zaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeMapByLS.zaxis.high = cms.untracked.double(30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeMapByLS.zaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeAll.xaxis.high = cms.untracked.double(30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeAll.xaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.Time1D.xaxis.high = cms.untracked.double(30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.Time1D.xaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeAllMap.zaxis.high = cms.untracked.double(25.)
      process.ecalMonitorTask.workerParameters.TimingTask.MEs.TimeAllMap.zaxis.low = cms.untracked.double(-25.)
      process.ecalMonitorTask.workerParameters.TimingTask.params.chi2ThresholdEE = cms.untracked.double(1000.)
      process.ecalMonitorTask.workerParameters.TimingTask.params.chi2ThresholdEB = cms.untracked.double(1000.)

      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdvBkwd.yaxis.high = cms.untracked.double(30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdvBkwd.yaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdvBkwd.xaxis.high = cms.untracked.double(30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdvBkwd.xaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdBkwdDiff.xaxis.high = cms.untracked.double(25.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.FwdBkwdDiff.xaxis.low = cms.untracked.double(-25.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.MeanSM.xaxis.high = cms.untracked.double(30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.MeanSM.xaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.MeanAll.xaxis.high = cms.untracked.double(30.)
      process.ecalMonitorClient.workerParameters.TimingClient.MEs.MeanAll.xaxis.low = cms.untracked.double(-30.)
      process.ecalMonitorClient.workerParameters.TimingClient.params.minChannelEntries = cms.untracked.int32(0)

process.ecalMonitorClient.verbosity = 0
process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'MLClient', 'SummaryClient']
process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
process.ecalMonitorClient.commonParameters.onlineMode = True

process.preScaler.prescaleFactor = 1

process.tcdsDigis = tcdsRawToDigi.clone(
  InputLabel = "rawDataCollector"
)

###### For OnlineLuminosityRecord to get the PU/luminosity info ######
process.load('EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi')
process.onlineMetaDataDigis = cms.EDProducer('OnlineMetaDataRawToDigi')

process.dqmEnv.subSystemFolder = 'Ecal'
process.dqmSaver.tag = 'Ecal'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'Ecal'
process.dqmSaverPB.runNumber = options.runNumber

process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"

process.ecalMonitorTask.workers = ['ClusterTask', 'EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask', 'PiZeroTask']
process.ecalMonitorTask.verbosity = 0
process.ecalMonitorTask.collectionTags.EESuperCluster = "multi5x5SuperClusters:multi5x5EndcapSuperClusters"
process.ecalMonitorTask.collectionTags.EBBasicCluster = "hybridSuperClusters:hybridBarrelBasicClusters"
process.ecalMonitorTask.collectionTags.EEBasicCluster = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
process.ecalMonitorTask.collectionTags.Source = "rawDataCollector"
process.ecalMonitorTask.collectionTags.EBSuperCluster = "correctedHybridSuperClusters"
process.ecalMonitorTask.collectionTags.TrigPrimEmulDigi = "simEcalTriggerPrimitiveDigis"
process.ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = True
process.ecalMonitorTask.commonParameters.willConvertToEDM = False
process.ecalMonitorTask.commonParameters.onlineMode = True
process.ecalMonitorTask.workerParameters.OccupancyTask.params.lumiCheck = True

### Sequences ###

process.ecalPreRecoSequence = cms.Sequence(process.bunchSpacingProducer + process.ecalDigis)
process.ecalRecoSequence = cms.Sequence((process.ecalMultiFitUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)+(process.simEcalTriggerPrimitiveDigis+process.gtDigis)+(process.hybridClusteringSequence+process.multi5x5ClusteringSequence))
process.multi5x5ClusteringSequence = cms.Sequence(process.multi5x5BasicClustersCleaned+process.multi5x5SuperClustersCleaned+process.multi5x5BasicClustersUncleaned+process.multi5x5SuperClustersUncleaned+process.multi5x5SuperClusters)
process.hybridClusteringSequence = cms.Sequence(process.cleanedHybridSuperClusters+process.uncleanedHybridSuperClusters+process.hybridSuperClusters+process.correctedHybridSuperClusters+process.uncleanedOnlyCorrectedHybridSuperClusters)

### Paths ###

process.ecalMonitorPath = cms.Path(process.onlineMetaDataDigis+process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalRecoSequence+process.tcdsDigis+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver + process.dqmSaverPB)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Run type specific ###

runTypeName = process.runType.getRunTypeName()
if (runTypeName == 'pp_run' or runTypeName == 'pp_run_stage1'):
    pass
elif (runTypeName == 'cosmic_run' or runTypeName == 'cosmic_run_stage1'):
#    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ['EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask', 'PiZeroTask']
    process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'MLClient', 'SummaryClient']
    process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
    process.ecalMonitorTask.workerParameters.PresampleTask.params.doPulseMaxCheck = False 
elif runTypeName == 'hi_run':
    process.ecalMonitorTask.collectionTags.Source = "rawDataRepacker"
    process.ecalDigis.cpu.InputLabel = 'rawDataRepacker'
elif runTypeName == 'hpu_run':
    if not unitTest:
        process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*'))


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
print("Final Source settings:", process.source)
process = customise(process)
