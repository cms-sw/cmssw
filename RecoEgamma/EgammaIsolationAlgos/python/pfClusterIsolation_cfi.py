import FWCore.ParameterSet.Config as cms

electronEcalPFClusterIsolationProducer = cms.EDProducer('ElectronEcalPFClusterIsolationProducer',
                                                        candidateProducer = cms.InputTag('gedGsfElectronsTmp'),
                                                        pfClusterProducer = cms.InputTag('particleFlowClusterECAL'),
                                                        drMax = cms.double(0.3),
                                                        drVetoBarrel = cms.double(0),
                                                        drVetoEndcap = cms.double(0),
                                                        etaStripBarrel = cms.double(0),
                                                        etaStripEndcap = cms.double(0),
                                                        energyBarrel = cms.double(0),
                                                        energyEndcap = cms.double(0)
                                                        )

photonEcalPFClusterIsolationProducer = cms.EDProducer('PhotonEcalPFClusterIsolationProducer',
                                                      candidateProducer = cms.InputTag('gedPhotonsTmp'),
                                                      pfClusterProducer = cms.InputTag('particleFlowClusterECAL'),
                                                      drMax = cms.double(0.3),
                                                      drVetoBarrel = cms.double(0),
                                                      drVetoEndcap = cms.double(0),
                                                      etaStripBarrel = cms.double(0),
                                                      etaStripEndcap = cms.double(0),
                                                      energyBarrel = cms.double(0),
                                                      energyEndcap = cms.double(0)
                                                      )

ootPhotonEcalPFClusterIsolationProducer = photonEcalPFClusterIsolationProducer.clone(
    candidateProducer = 'ootPhotonsTmp',
    pfClusterProducer = 'particleFlowClusterOOTECAL'
)
electronHcalPFClusterIsolationProducer = cms.EDProducer('ElectronHcalPFClusterIsolationProducer',
                                                        candidateProducer = cms.InputTag('gedGsfElectronsTmp'),
                                                        pfClusterProducerHCAL = cms.InputTag('particleFlowClusterHCAL'),
                                                        useHF = cms.bool(False),
                                                        drMax = cms.double(0.3),
                                                        drVetoBarrel = cms.double(0),
                                                        drVetoEndcap = cms.double(0),
                                                        etaStripBarrel = cms.double(0),
                                                        etaStripEndcap = cms.double(0),
                                                        energyBarrel = cms.double(0),
                                                        energyEndcap = cms.double(0)
                                                        )

photonHcalPFClusterIsolationProducer = cms.EDProducer('PhotonHcalPFClusterIsolationProducer',
                                                      candidateProducer = cms.InputTag('gedPhotonsTmp'),
                                                      pfClusterProducerHCAL = cms.InputTag('particleFlowClusterHCAL'),
                                                      useHF = cms.bool(False),
                                                      drMax = cms.double(0.3),
                                                      drVetoBarrel = cms.double(0),
                                                      drVetoEndcap = cms.double(0),
                                                      etaStripBarrel = cms.double(0),
                                                      etaStripEndcap = cms.double(0),
                                                      energyBarrel = cms.double(0),
                                                      energyEndcap = cms.double(0)
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
