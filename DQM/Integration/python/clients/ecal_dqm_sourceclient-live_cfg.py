### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
import FWCore.ParameterSet.Config as cms

process = cms.Process("process")

### Load cfis ###

process.load("DQM.Integration.config.inputsource_cfi")
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

process.ecalMonitorClient.verbosity = 0
process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
process.ecalMonitorClient.commonParameters.onlineMode = True

process.GlobalTag.toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalDQMChannelStatusRcd'),
    tag = cms.string('EcalDQMChannelStatus_v1_hlt'),
), 
    cms.PSet(
        record = cms.string('EcalDQMTowerStatusRcd'),
        tag = cms.string('EcalDQMTowerStatus_v1_hlt'),
    ))

process.preScaler.prescaleFactor = 1

process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference.root"

process.dqmEnv.subSystemFolder = cms.untracked.string('Ecal')
process.dqmSaver.tag = cms.untracked.string('Ecal')

process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"

process.ecalMonitorTask.workers = ['ClusterTask', 'EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
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

### Sequences ###

process.ecalPreRecoSequence = cms.Sequence(process.bunchSpacingProducer + process.ecalDigis)
process.ecalRecoSequence = cms.Sequence((process.ecalMultiFitUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)+(process.simEcalTriggerPrimitiveDigis+process.gtDigis)+(process.hybridClusteringSequence+process.multi5x5ClusteringSequence))
process.multi5x5ClusteringSequence = cms.Sequence(process.multi5x5BasicClustersCleaned+process.multi5x5SuperClustersCleaned+process.multi5x5BasicClustersUncleaned+process.multi5x5SuperClustersUncleaned+process.multi5x5SuperClusters)
process.hybridClusteringSequence = cms.Sequence(process.cleanedHybridSuperClusters+process.uncleanedHybridSuperClusters+process.hybridSuperClusters+process.correctedHybridSuperClusters+process.uncleanedOnlyCorrectedHybridSuperClusters)

### Paths ###

process.ecalMonitorPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalRecoSequence+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Run type specific ###

referenceFileName = process.DQMStore.referenceFileName.pythonValue()
runTypeName = process.runType.getRunTypeName()
if (runTypeName == 'pp_run' or runTypeName == 'pp_run_stage1'):
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_pp.root')
elif (runTypeName == 'cosmic_run' or runTypeName == 'cosmic_run_stage1'):
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_cosmic.root')
#    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ['EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
    process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
    process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
    process.ecalMonitorTask.workerParameters.PresampleTask.params.doPulseMaxCheck = False 
elif runTypeName == 'hi_run':
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hi.root')
    process.ecalMonitorTask.collectionTags.Source = "rawDataRepacker"
    process.ecalDigis.InputLabel = cms.InputTag('rawDataRepacker')
elif runTypeName == 'hpu_run':
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hpu.root')
    process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*'))


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
