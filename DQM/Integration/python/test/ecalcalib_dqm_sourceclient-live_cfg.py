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

from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import ecalFixedAlphaBetaFitUncalibRecHit
process.ecalLaserLedUncalibRecHit = ecalFixedAlphaBetaFitUncalibRecHit.clone()

from RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi import ecalMaxSampleUncalibRecHit
process.ecalTestPulseUncalibRecHit = ecalMaxSampleUncalibRecHit.clone()

    
### ECAL DQM MODULES ###

process.load("DQM.EcalCommon.EcalDQMBinningService_cfi")

process.load("DQM.EcalBarrelMonitorTasks.EcalCalibMonitorTasks_cfi")
process.load("DQM.EcalBarrelMonitorClient.EcalCalibMonitorClient_cfi")


### DQM COMMON MODULES ###

process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.Integration.test.environment_cfi")


### FILTERS ###

process.load("FWCore.Modules.preScaler_cfi")

process.ecalCalibrationFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalLaserLedFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler")


### JOB PARAMETERS ###

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

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

process.ecalLaserLedPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalLaserLedFilter *    
    process.ecalRecoSequence *
    process.ecalLaserLedUncalibRecHit *
    process.ecalLaserLedMonitorTask *
    process.ecalPNDiodeMonitorTask
)
process.ecalTestPulsePath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalTestPulseFilter *    
    process.ecalRecoSequence *
    process.ecalTestPulseUncalibRecHit *
    process.ecalTestPulseMonitorTask *
    process.ecalPNDiodeMonitorTask
)

process.ecalClientPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalCalibrationFilter *
    process.ecalCalibMonitorClient
)

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.ecalLaserLedPath,
    process.ecalTestPulsePath,
    process.ecalClientPath,
    process.dqmEndPath
)


### SOURCE ###

process.load("DQM.Integration.test.inputsource_cfi")


### CUSTOMIZATIONS ###

 ## Reconstruction Modules ##

process.ecalRecHit.killDeadChannels = True
process.ecalRecHit.ChannelStatusToBeExcluded = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

process.ecalTestPulseUncalibRecHit.EBdigiCollection = "ecalDigis:ebDigis"
process.ecalTestPulseUncalibRecHit.EEdigiCollection = "ecalDigis:eeDigis"
    
process.ecalLaserLedUncalibRecHit.MinAmplBarrel = 12.
process.ecalLaserLedUncalibRecHit.MinAmplEndcap = 16.

 ## Filters ##

process.ecalCalibrationFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalCalibrationFilter.laserPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.ledPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.pedestalPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.testpulsePrescaleFactor = cms.untracked.int32(1)

process.ecalLaserLedFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalLaserLedFilter.laserPrescaleFactor = cms.untracked.int32(1)
process.ecalLaserLedFilter.ledPrescaleFactor = cms.untracked.int32(1)

process.ecalTestPulseFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalTestPulseFilter.testpulsePrescaleFactor = cms.untracked.int32(1)

 ## Ecal DQM modules ##

process.ecalLaserLedMonitorTask.workerParameters.common.laserWavelengths = [2, 3, 4]
process.ecalCalibMonitorClient.workerParameters.common.laserWavelengths = [2, 3, 4]

 ## DQM common modules ##

process.dqmEnv.subSystemFolder = cms.untracked.string("EcalCalibration")
process.dqmSaver.convention = "Online"

 ## Source ##
process.source.consumerName = cms.untracked.string("EcalCalibration DQM Consumer")
process.source.SelectHLTOutput = cms.untracked.string("hltOutputCalibration")
process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("HLT_EcalCalibration_v*"))

 ## Run type specific ##

 ## FEDRawDataCollection name ##
FedRawData = "hltEcalCalibrationRaw"

process.ecalDigis.InputLabel = cms.InputTag(FedRawData)
