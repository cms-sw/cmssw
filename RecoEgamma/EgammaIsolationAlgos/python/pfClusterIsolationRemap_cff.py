import FWCore.ParameterSet.Config as cms

electronHcalPFClusterIsolationRemapper = cms.EDProducer('ElectronPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGsfElectrons'),
    newToOldObjectMap = cms.InputTag('gsFixedGsfElectrons'),
    isolationMap = cms.InputTag('electronHcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

photonHcalPFClusterIsolationRemapper = cms.EDProducer('PhotonPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGEDPhotons'),
    newToOldObjectMap = cms.InputTag('gsFixedGEDPhotons'),
    isolationMap = cms.InputTag('photonHcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

electronEcalPFClusterIsolationRemapper = cms.EDProducer('ElectronPFClusterIsolationRemapper',
    candidateProducer = cms.InputTag('gsFixedGsfElectrons'),
    newToOldObjectMap = cms.InputTag('gsFixedGsfElectrons'),
    isolationMap = cms.InputTag('electronEcalPFClusterIsolationProducer', '', cms.InputTag.skipCurrentProcess())
)

photonEcalPFClusterIsolationRemapper = cms.EDProducer('PhotonPFClusterIsolationRemapper',
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
