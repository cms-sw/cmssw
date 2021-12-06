import FWCore.ParameterSet.Config as cms
import RecoEgamma.EgammaIsolationAlgos.egammaEcalPFClusterIsolationProducerRecoGsfElectron_cfi as _mod_Ecalele
import RecoEgamma.EgammaIsolationAlgos.egammaEcalPFClusterIsolationProducerRecoPhoton_cfi as _mod_Ecalpho  
import RecoEgamma.EgammaIsolationAlgos.egammaHcalPFClusterIsolationProducerRecoGsfElectron_cfi as _mod_Hcalele
import RecoEgamma.EgammaIsolationAlgos.egammaHcalPFClusterIsolationProducerRecoPhoton_cfi as _mod_Hcalpho

electronEcalPFClusterIsolationProducer = _mod_Ecalele.egammaEcalPFClusterIsolationProducerRecoGsfElectron.clone(
                                                      candidateProducer = 'gedGsfElectronsTmp',
                                                      )

photonEcalPFClusterIsolationProducer = _mod_Ecalpho.egammaEcalPFClusterIsolationProducerRecoPhoton.clone(
                                                      candidateProducer = 'gedPhotonsTmp',
                                                      )

ootPhotonEcalPFClusterIsolationProducer = photonEcalPFClusterIsolationProducer.clone(
    candidateProducer = 'ootPhotonsTmp',
    pfClusterProducer = 'particleFlowClusterOOTECAL'
)

electronHcalPFClusterIsolationProducer = _mod_Hcalele.egammaHcalPFClusterIsolationProducerRecoGsfElectron.clone( 
                                                      candidateProducer = 'gedGsfElectronsTmp',
                                                      )

photonHcalPFClusterIsolationProducer = _mod_Hcalpho.egammaHcalPFClusterIsolationProducerRecoPhoton.clone( 
                                                      candidateProducer = 'gedPhotonsTmp',
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
