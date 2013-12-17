import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

mustachePhotonCore = photonCore.clone()
mustachePhotonCore.scHybridBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
mustachePhotonCore.scIslandEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
mustachePhotonCore.conversionProducer = cms.InputTag('conversions')
mustachePhotons = photons.clone()
mustachePhotons.photonCoreProducer = cms.InputTag('mustachePhotonCore')
mustachePhotonSequence = cms.Sequence( mustachePhotonCore + mustachePhotons )

photonSequence = cms.Sequence( photonCore + photons + mustachePhotonSequence )

