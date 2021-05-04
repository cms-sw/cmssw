import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *

ootPhotonCore = photonCore.clone(
    scHybridBarrelProducer = "particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALBarrel",
    scIslandEndcapProducer = "particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALEndcapWithPreshower",
    conversionProducer     = "conversions"
)
