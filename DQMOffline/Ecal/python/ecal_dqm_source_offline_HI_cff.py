import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.EcalFEDMonitor_cfi import *
from DQMOffline.Ecal.EcalZmassTask_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoEcal = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('Ecal')
)

ecalMultiftAnalyzer = DQMEDAnalyzer('ECALMultifitAnalyzer_HI',
                                     recoPhotonSrc         = cms.InputTag('photons'),
                                     recoJetSrc            = cms.InputTag('akPu4CaloJets'),
                                     RecHitCollection_EB   = cms.InputTag('ecalRecHit:EcalRecHitsEB'),
                                     RecHitCollection_EE   = cms.InputTag('ecalRecHit:EcalRecHitsEE'),
                                     rechitEnergyThreshold = cms.double(5.0),
                                     recoPhotonPtThreshold = cms.double(15.0),
                                     recoJetPtThreshold    = cms.double(30.0),
                                     deltaRPhotonThreshold = cms.double(0.1),
                                     deltaRJetThreshold    = cms.double(0.4)
)

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal +
    ecalMonitorTask +
    ecalFEDMonitor +
    ecalzmasstask +
    ecalMultiftAnalyzer
)

ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = False
ecalMonitorTask.collectionTags.Source = 'rawDataMapperByLabel'
ecalMonitorTask.collectionTags.EBBasicCluster = 'islandBasicClusters:islandBarrelBasicClusters'
ecalMonitorTask.collectionTags.EEBasicCluster = 'islandBasicClusters:islandEndcapBasicClusters'
ecalMonitorTask.collectionTags.EBSuperCluster = 'correctedIslandBarrelSuperClusters'
ecalMonitorTask.collectionTags.EESuperCluster = 'correctedIslandEndcapSuperClusters'
