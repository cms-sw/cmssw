import FWCore.ParameterSet.Config as cms

import DQMOffline.EGamma.photonAnalyzer_cfi

cosmicPhotonAnalysis =  DQMOffline.EGamma.photonAnalyzer_cfi.photonAnalysis.clone(
    ComponentName = 'cosmicPhotonAnalysis',
    analyzerName = 'stdPhotonAnalyzer',
    phoProducer = 'photons',
    minPhoEtCut = 0.0,
    eMax = 3.0,
    etMax = 3.0,
    r9Max = 1.5,    
    
    barrelRecHitProducer = 'ecalRecHit',
    endcapRecHitProducer = 'ecalRecHit'
)
