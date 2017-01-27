import FWCore.ParameterSet.Config as cms

electronHcalPFClusterIsolationRemapper = cms.EDProducer('ElectronHcalPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGsfElectrons'),
    newToOldObjectMap = cms.InputTag('gsFixedGsfElectrons'),
    isolationMap = cms.InputTag('electronHcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

photonHcalPFClusterIsolationRemapper = cms.EDProducer('PhotonHcalPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGEDPhotons'),
    newToOldObjectMap = cms.InputTag('gsFixedGEDPhotons'),
    isolationMap = cms.InputTag('photonHcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

electronEcalPFClusterIsolationRemapper = cms.EDProducer('ElectronEcalPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGsfElectrons'),
    newToOldObjectMap = cms.InputTag('gsFixedGsfElectrons'),
    isolationMap = cms.InputTag('electronEcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

photonEcalPFClusterIsolationRemapper = cms.EDProducer('PhotonEcalPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGEDPhotons'),
    newToOldObjectMap = cms.InputTag('gsFixedGEDPhotons'),
    isolationMap = cms.InputTag('photonEcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

pfClusterIsolationRemapSequence = cms.Sequence(
    electronHcalPFClusterIsolationRemapper +
    photonHcalPFClusterIsolationRemapper + 
    electronEcalPFClusterIsolationRemapper +
    photonEcalPFClusterIsolationRemapper
)
