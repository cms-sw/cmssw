import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaHBHERecHitThreshold_cff import egammaHBHERecHit

from RecoEgamma.EgammaIsolationAlgos.egammaEcalPFClusterIsolationProducerRecoGsfElectron_cfi import egammaEcalPFClusterIsolationProducerRecoGsfElectron
from RecoEgamma.EgammaIsolationAlgos.egammaHcalPFClusterIsolationProducerRecoGsfElectron_cfi import egammaHcalPFClusterIsolationProducerRecoGsfElectron 

import RecoEgamma.EgammaElectronProducers.gsfElectronProducerDefault_cfi as _gsfProd
gsfElectronProducer = _gsfProd.gsfElectronProducerDefault.clone(
    hbheRecHits = egammaHBHERecHit.hbheRecHits,
    recHitEThresholdHB = egammaHBHERecHit.recHitEThresholdHB,
    recHitEThresholdHE = egammaHBHERecHit.recHitEThresholdHE,
    usePFThresholdsFromDB = egammaHBHERecHit.usePFThresholdsFromDB,
    maxHcalRecHitSeverity = egammaHBHERecHit.maxHcalRecHitSeverity,

    pfECALClusIsolCfg = cms.PSet( 
        pfClusterProducer = egammaEcalPFClusterIsolationProducerRecoGsfElectron.pfClusterProducer,
        drMax = egammaEcalPFClusterIsolationProducerRecoGsfElectron.drMax,
        drVetoBarrel = egammaEcalPFClusterIsolationProducerRecoGsfElectron.drVetoBarrel,
        drVetoEndcap = egammaEcalPFClusterIsolationProducerRecoGsfElectron.drVetoEndcap,
        etaStripBarrel = egammaEcalPFClusterIsolationProducerRecoGsfElectron.etaStripBarrel,
        etaStripEndcap = egammaEcalPFClusterIsolationProducerRecoGsfElectron.etaStripEndcap,
        energyBarrel = egammaEcalPFClusterIsolationProducerRecoGsfElectron.energyBarrel,
        energyEndcap = egammaEcalPFClusterIsolationProducerRecoGsfElectron.energyEndcap
    ),

    pfHCALClusIsolCfg = cms.PSet(

        pfClusterProducerHCAL = egammaHcalPFClusterIsolationProducerRecoGsfElectron.pfClusterProducerHCAL,
        useHF = egammaHcalPFClusterIsolationProducerRecoGsfElectron.useHF,
        pfClusterProducerHFEM = egammaHcalPFClusterIsolationProducerRecoGsfElectron.pfClusterProducerHFEM,
        pfClusterProducerHFHAD = egammaHcalPFClusterIsolationProducerRecoGsfElectron.pfClusterProducerHFHAD,
        drMax = egammaHcalPFClusterIsolationProducerRecoGsfElectron.drMax,
        drVetoBarrel = egammaHcalPFClusterIsolationProducerRecoGsfElectron.drVetoBarrel,
        drVetoEndcap = egammaHcalPFClusterIsolationProducerRecoGsfElectron.drVetoEndcap,
        etaStripBarrel = egammaHcalPFClusterIsolationProducerRecoGsfElectron.etaStripBarrel,
        etaStripEndcap = egammaHcalPFClusterIsolationProducerRecoGsfElectron.etaStripEndcap,
        energyBarrel = egammaHcalPFClusterIsolationProducerRecoGsfElectron.energyBarrel,
        energyEndcap = egammaHcalPFClusterIsolationProducerRecoGsfElectron.energyEndcap,
        useEt = egammaHcalPFClusterIsolationProducerRecoGsfElectron.useEt,

    )

)
