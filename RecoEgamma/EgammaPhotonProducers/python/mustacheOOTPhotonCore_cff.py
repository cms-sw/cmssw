import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *

mustacheOOTPhotonCore = photonCore.clone()
mustacheOOTPhotonCore.scHybridBarrelProducer = cms.InputTag("particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALBarrel")
mustacheOOTPhotonCore.scIslandEndcapProducer = cms.InputTag("particleFlowSuperClusterOOTECAL:particleFlowSuperClusterOOTECALEndcapWithPreshower")
mustacheOOTPhotonCore.conversionProducer = cms.InputTag("conversions")

