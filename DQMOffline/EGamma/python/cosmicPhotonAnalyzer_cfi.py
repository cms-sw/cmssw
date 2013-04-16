import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import photonAnalysis as _photonAnalysis 
cosmicPhotonAnalysis = _photonAnalysis.clone()

cosmicPhotonAnalysis.minPhoEtCut = cms.double(0.0)
cosmicPhotonAnalysis.eMax = cms.double(3.0)
cosmicPhotonAnalysis.etMax = cms.double(3.0)
cosmicPhotonAnalysis.r9Max = cms.double(1.5)    

cosmicPhotonAnalysis.barrelRecHitProducer = cms.string('ecalRecHit')
cosmicPhotonAnalysis.barrelRecHitCollection = cms.string('EcalRecHitsEB')
cosmicPhotonAnalysis.endcapRecHitProducer = cms.string('ecalRecHit')
cosmicPhotonAnalysis.endcapRecHitCollection = cms.string('EcalRecHitsEE')
