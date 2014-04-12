import FWCore.ParameterSet.Config as cms

import DQMOffline.EGamma.photonAnalyzer_cfi

cosmicPhotonAnalysis =  DQMOffline.EGamma.photonAnalyzer_cfi.photonAnalysis.clone()
cosmicPhotonAnalysis.ComponentName = cms.string('cosmicPhotonAnalysis')
cosmicPhotonAnalysis.analyzerName = cms.string('stdPhotonAnalyzer')
cosmicPhotonAnalysis.phoProducer = cms.InputTag('photons')
cosmicPhotonAnalysis.minPhoEtCut = cms.double(0.0)
cosmicPhotonAnalysis.eMax = cms.double(3.0)
cosmicPhotonAnalysis.etMax = cms.double(3.0)
cosmicPhotonAnalysis.r9Max = cms.double(1.5)    

cosmicPhotonAnalysis.barrelRecHitProducer = cms.InputTag('ecalRecHit')
cosmicPhotonAnalysis.barrelRecHitCollection = cms.InputTag('EcalRecHitsEB')
cosmicPhotonAnalysis.endcapRecHitProducer = cms.InputTag('ecalRecHit')
cosmicPhotonAnalysis.endcapRecHitCollection = cms.InputTag('EcalRecHitsEE')
