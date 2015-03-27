import FWCore.ParameterSet.Config as cms

electronEcalPFClusterIsolationProducer = cms.EDProducer('ElectronEcalPFClusterIsolationProducer',
                                                        candidateProducer = cms.InputTag('gedGsfElectrons'),
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
                                                      candidateProducer = cms.InputTag('gedPhotons'),
                                                      pfClusterProducer = cms.InputTag('particleFlowClusterECAL'),
                                                      drMax = cms.double(0.3),
                                                      drVetoBarrel = cms.double(0),
                                                      drVetoEndcap = cms.double(0),
                                                      etaStripBarrel = cms.double(0),
                                                      etaStripEndcap = cms.double(0),
                                                      energyBarrel = cms.double(0),
                                                      energyEndcap = cms.double(0)
                                                      )

electronHcalPFClusterIsolationProducer = cms.EDProducer('ElectronHcalPFClusterIsolationProducer',
                                                        candidateProducer = cms.InputTag('gedGsfElectrons'),
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
                                                      candidateProducer = cms.InputTag('gedPhotons'),
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

pfClusterIsolationSequence = cms.Sequence(
    electronEcalPFClusterIsolationProducer *
    photonEcalPFClusterIsolationProducer *
    electronHcalPFClusterIsolationProducer *
    photonHcalPFClusterIsolationProducer 
)
