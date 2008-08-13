import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import photonAnalysis as cosmicPhotonAnalysis 
cosmicPhotonAnalysis.bcBarrelCollection = cms.InputTag("cosmicBasicClusters","CosmicBarrelBasicClusters")
cosmicPhotonAnalysis.bcEndcapCollection = cms.InputTag("cosmicBasicClusters","CosmicEndcapBasicClusters")
cosmicPhotonAnalysis.scBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
cosmicPhotonAnalysis.scEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
cosmicPhotonAnalysis.minPhoEtCut = cms.double(0.0)
cosmicPhotonAnalysis.eMax = cms.double(1.0)
cosmicPhotonAnalysis.etMax = cms.double(1.0)

