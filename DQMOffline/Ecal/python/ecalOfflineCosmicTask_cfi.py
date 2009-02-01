import FWCore.ParameterSet.Config as cms

ecalOfflineCosmicTask = cms.EDAnalyzer("EcalOfflineCosmicTask",
    histogramMinRange = cms.untracked.double(0.0),
    L1GlobalMuonReadoutRecord = cms.untracked.string('gtDigis'),
    ecalRecHitCollectionEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    ecalRecHitCollectionEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    superClusterCollectionEB = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters"),
    superClusterCollectionEE = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters"),
    L1GlobalReadoutRecord = cms.untracked.string('gtDigis'),
    histogramMaxRange = cms.untracked.double(1.8),
    fileName = cms.untracked.string('EcalOfflineCosmicTask.root'),
    saveFile = cms.untracked.bool(False),
#    MinTimingAmpEB = cms.untracked.double(0.1),   # for adcToGeV=0.009, gain 200
#    MinTimingAmpEB = cms.untracked.double(0.35),  # for adcToGeV=0.035, gain 50
    MinTimingAmpEE = cms.untracked.double(0.9),   # for adcToGeV=0.06
    MinHighEnergy = cms.untracked.double(2.0)
)


