import FWCore.ParameterSet.Config as cms

#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
allConversionSequence = cms.Sequence(allConversions)

allConversionsMustache = allConversions.clone()
allConversionsMustache.scBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
allConversionsMustache.bcBarrelCollection = cms.InputTag('particleFlowClusterECAL')
allConversionsMustache.scEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
allConversionsMustache.bcEndcapCollection = cms.InputTag('particleFlowClusterECAL')

allConversionMustacheSequence = cms.Sequence(allConversionsMustache)

