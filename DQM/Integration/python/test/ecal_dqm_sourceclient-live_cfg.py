import FWCore.ParameterSet.Config as cms
import os, sys, socket

process = cms.Process("DQM")


### RECONSTRUCTION MODULES ###

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker
process.ecalDigis = ecalEBunpacker.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")

    
### ECAL DQM MODULES ###

process.load("DQM.EcalCommon.EcalDQMBinningService_cfi")

process.load("DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalBarrelMonitorClient.EcalMonitorClient_cfi")

### DQM COMMON MODULES ###


process.dqmQTest = cms.EDAnalyzer("QualityTester",
    reportThreshold = cms.untracked.string("red"),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath("DQM/EcalCommon/data/EcalQualityTests.xml"),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.load("DQM.Integration.test.environment_cfi")


### FILTERS ###

process.load("FWCore.Modules.preScaler_cfi")

process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler")



### JOB PARAMETERS ###

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# Condition for P5 cluster
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
# Condition for lxplus
#process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string("EcalDQMChannelStatusRcd"),
        tag = cms.string("EcalDQMChannelStatus_v1_hlt"),
        connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL")
    ),
    cms.PSet(
        record = cms.string("EcalDQMTowerStatusRcd"),
        tag = cms.string("EcalDQMTowerStatus_v1_hlt"),
        connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL")
    )
)


### MESSAGE LOGGER ###

process.MessageLogger = cms.Service("MessageLogger",
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string("WARNING"),
    noLineBreaks = cms.untracked.bool(True),
    noTimeStamps = cms.untracked.bool(True),
    default = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    )
  ),
  destinations = cms.untracked.vstring("cout")
)


### SEQUENCES AND PATHS ###

process.ecalPreRecoSequence = cms.Sequence(
    process.preScaler +
    process.ecalDigis
)

process.ecalRecoSequence = cms.Sequence(
    process.ecalGlobalUncalibRecHit *
    process.ecalDetIdToBeRecovered *
    process.ecalRecHit
)

process.ecalClusterSequence = cms.Sequence(
    process.hybridClusteringSequence *
    process.multi5x5ClusteringSequence
)
process.ecalClusterSequence.remove(process.multi5x5SuperClustersWithPreshower)

process.ecalMonitorPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalPhysicsFilter *
    process.ecalRecoSequence *
    process.ecalClusterSequence *
    process.simEcalTriggerPrimitiveDigis *
    process.ecalMonitorTask
)

process.ecalClientPath = cms.Path(
    process.ecalMonitorClient
)

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmQTest *
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.ecalMonitorPath,
    process.ecalClientPath,
    process.dqmEndPath
)


### SOURCE ###

process.load("DQM.Integration.test.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

### CUSTOMIZATIONS ###

 ## Reconstruction Modules ##

process.ecalRecHit.killDeadChannels = True
process.ecalRecHit.ChannelStatusToBeExcluded = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"

 ## Filters ##

process.ecalPhysicsFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalPhysicsFilter.clusterPrescaleFactor = cms.untracked.int32(1)

 ## Ecal DQM modules ##

process.ecalMonitorTask.online = True
process.ecalMonitorTask.workers = ["ClusterTask", "EnergyTask", "IntegrityTask", "OccupancyTask", "RawDataTask", "TimingTask", "TrigPrimTask", "PresampleTask", "SelectiveReadoutTask"]
process.ecalMonitorTask.workerParameters.common.hltTaskMode = 0
process.ecalMonitorTask.workerParameters.TrigPrimTask.runOnEmul = True

process.ecalMonitorClient.online = True
process.ecalMonitorClient.workers = ["IntegrityClient", "OccupancyClient", "PresampleClient", "RawDataClient", "TimingClient", "SelectiveReadoutClient", "TrigPrimClient", "SummaryClient"]
process.ecalMonitorClient.workerParameters.SummaryClient.activeSources = ["Integrity", "RawData", "Presample", "TriggerPrimitives", "Timing", "HotCell"]

 ## DQM common modules ##

process.dqmEnv.subSystemFolder = cms.untracked.string("Ecal")
process.dqmSaver.convention = "Online"

 ## Run type specific ##

if process.runType.getRunType() == process.runType.pp_run :
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference_pp.root"
elif process.runType.getRunType() == process.runType.cosmic_run :
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference_cosmic.root"
    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ["EnergyTask", "IntegrityTask", "OccupancyTask", "RawDataTask", "TrigPrimTask", "PresampleTask", "SelectiveReadoutTask"]
    process.ecalMonitorClient.workers = ["IntegrityClient", "OccupancyClient", "PresampleClient", "RawDataClient", "SelectiveReadoutClient", "TrigPrimClient", "SummaryClient"]
    process.ecalMonitorClient.workerParameters.SummaryClient.activeSources = ["Integrity", "RawData", "Presample", "TriggerPrimitives", "HotCell"]
elif process.runType.getRunType() == process.runType.hi_run:
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference_hi.root"
elif process.runType.getRunType() == process.runType.hpu_run:
    process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("*"))

 ## FEDRawDataCollection name ##
FedRawData = "rawDataCollector"

if process.runType.getRunType() == process.runType.hi_run:
    FedRawData = "rawDataRepacker"

process.ecalDigis.InputLabel = cms.InputTag(FedRawData)
process.ecalMonitorTask.collectionTags.Source = FedRawData
