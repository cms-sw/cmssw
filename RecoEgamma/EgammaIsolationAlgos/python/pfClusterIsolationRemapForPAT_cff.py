import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemap_cff import electronHcalPFClusterIsolationRemapper as _electronHcalPFClusterIsolationRemapper
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemap_cff import electronEcalPFClusterIsolationRemapper as _electronEcalPFClusterIsolationRemapper
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemap_cff import photonHcalPFClusterIsolationRemapper as _photonHcalPFClusterIsolationRemapper
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemap_cff import photonEcalPFClusterIsolationRemapper as _photonEcalPFClusterIsolationRemapper

electronHcalPFClusterIsolationProducerGSFixed = _electronHcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedGsfElectronsGSFixed',
    newToOldObjectMap = 'gedGsfElectronsGSFixed'
)

photonHcalPFClusterIsolationProducerGSFixed = _photonHcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedPhotonsGSFixed',
    newToOldObjectMap = 'gedPhotonsGSFixed'
)

electronEcalPFClusterIsolationProducerGSFixed = _electronEcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedGsfElectronsGSFixed',
    newToOldObjectMap = 'gedGsfElectronsGSFixed'
)

photonEcalPFClusterIsolationProducerGSFixed = _photonEcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedPhotonsGSFixed',
    newToOldObjectMap = 'gedPhotonsGSFixed'
)

pfClusterIsolationSequence = cms.Sequence(
    electronEcalPFClusterIsolationProducerGSFixed +
    photonEcalPFClusterIsolationProducerGSFixed +
    electronHcalPFClusterIsolationProducerGSFixed +
    photonHcalPFClusterIsolationProducerGSFixed
)
