import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *

ootPhotonCore = photonCore.clone()
ootPhotonCore.scHybridBarrelProducer = cms.InputTag("particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALBarrel")
ootPhotonCore.scIslandEndcapProducer = cms.InputTag("particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALEndcapWithPreshower")
ootPhotonCore.conversionProducer = cms.InputTag("conversions")
