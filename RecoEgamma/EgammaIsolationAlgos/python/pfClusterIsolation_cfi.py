import FWCore.ParameterSet.Config as cms
import RecoEgamma.EgammaIsolationAlgos.egammaEcalPFClusterIsolationProducerRecoGsfElectron_cfi as _mod_Ecalele
import RecoEgamma.EgammaIsolationAlgos.egammaEcalPFClusterIsolationProducerRecoPhoton_cfi as _mod_Ecalpho  
import RecoEgamma.EgammaIsolationAlgos.egammaHcalPFClusterIsolationProducerRecoGsfElectron_cfi as _mod_Hcalele
import RecoEgamma.EgammaIsolationAlgos.egammaHcalPFClusterIsolationProducerRecoPhoton_cfi as _mod_Hcalpho

electronEcalPFClusterIsolationProducer = _mod_Ecalele.egammaEcalPFClusterIsolationProducerRecoGsfElectron.clone(
                                                      candidateProducer = 'gedGsfElectronsTmp',
                                                      pfClusterProducer = 'particleFlowClusterECAL',
                                                      drMax = 0.3,
                                                      drVetoBarrel = 0,
                                                      drVetoEndcap = 0,
                                                      etaStripBarrel = 0,
                                                      etaStripEndcap = 0,
                                                      energyBarrel = 0,
                                                      energyEndcap = 0
                                                      )

photonEcalPFClusterIsolationProducer = _mod_Ecalpho.egammaEcalPFClusterIsolationProducerRecoPhoton.clone(
                                                      candidateProducer = 'gedPhotonsTmp',
                                                      pfClusterProducer = 'particleFlowClusterECAL',
                                                      drMax = 0.3,
                                                      drVetoBarrel = 0,
                                                      drVetoEndcap = 0,
                                                      etaStripBarrel = 0,
                                                      etaStripEndcap = 0,
                                                      energyBarrel = 0,
                                                      energyEndcap = 0
                                                      )

ootPhotonEcalPFClusterIsolationProducer = photonEcalPFClusterIsolationProducer.clone(
    candidateProducer = 'ootPhotonsTmp',
    pfClusterProducer = 'particleFlowClusterOOTECAL'
)

electronHcalPFClusterIsolationProducer = _mod_Hcalele.egammaHcalPFClusterIsolationProducerRecoGsfElectron.clone( 
                                                      candidateProducer = 'gedGsfElectronsTmp',
                                                      pfClusterProducerHCAL = 'particleFlowClusterHCAL',
                                                      useHF = False,
                                                      drMax = 0.3,
                                                      drVetoBarrel = 0,
                                                      drVetoEndcap = 0,
                                                      etaStripBarrel = 0,
                                                      etaStripEndcap = 0,
                                                      energyBarrel = 0,
                                                      energyEndcap = 0
                                                      )

photonHcalPFClusterIsolationProducer = _mod_Hcalpho.egammaHcalPFClusterIsolationProducerRecoPhoton.clone( 
                                                      candidateProducer = 'gedPhotonsTmp',
                                                      pfClusterProducerHCAL = 'particleFlowClusterHCAL',
                                                      useHF = False,
                                                      drMax = 0.3,
                                                      drVetoBarrel = 0,
                                                      drVetoEndcap = 0,
                                                      etaStripBarrel = 0,
                                                      etaStripEndcap = 0,
                                                      energyBarrel = 0,
                                                      energyEndcap = 0
                                                      )

ootPhotonHcalPFClusterIsolationProducer = photonHcalPFClusterIsolationProducer.clone(
    candidateProducer = 'ootPhotonsTmp'
)
pfClusterIsolationTask = cms.Task(
    electronEcalPFClusterIsolationProducer ,
    photonEcalPFClusterIsolationProducer ,
    ootPhotonEcalPFClusterIsolationProducer ,
    electronHcalPFClusterIsolationProducer ,
    photonHcalPFClusterIsolationProducer ,
    ootPhotonHcalPFClusterIsolationProducer 
)
pfClusterIsolationSequence = cms.Sequence(pfClusterIsolationTask)
