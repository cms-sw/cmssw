import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolation_cfi import *
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemap_cfi import *

electronHcalPFClusterIsolationProducerGSFixed = electronHcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedGsfElectronsGSFixed',
    newToOldObjectMap = 'gedGsfElectronsGSFixed'
)

photonHcalPFClusterIsolationProducerGSFixed = photonHcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedPhotonsGSFixed',
    newToOldObjectMap = 'gedPhotonsGSFixed'
)

electronEcalPFClusterIsolationProducerGSFixed = electronEcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedGsfElectronsGSFixed',
    newToOldObjectMap = 'gedGsfElectronsGSFixed'
)

photonEcalPFClusterIsolationProducerGSFixed = photonEcalPFClusterIsolationRemapper.clone(
    candidateProducer = 'gedPhotonsGSFixed',
    newToOldObjectMap = 'gedPhotonsGSFixed'
)

pfClusterIsolationSequence = cms.Sequence(
    electronEcalPFClusterIsolationProducerGSFixed +
    photonEcalPFClusterIsolationProducerGSFixed +
    electronHcalPFClusterIsolationProducerGSFixed +
    photonHcalPFClusterIsolationProducerGSFixed
)
