import FWCore.ParameterSet.Config as cms

#
#
# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTracks_cff import *
# converted photon producer
#from RecoEgamma.EgammaTools.PhotonConversionMVAComputer_cfi import *
from RecoEgamma.EgammaPhotonProducers.conversions_cfi import *
#conversionSequence = cms.Sequence(ckfTracksFromConversions*conversions)
conversionSequence = cms.Sequence(conversions)

mustacheConversions = conversions.clone()
mustacheConversions.scHybridBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
mustacheConversions.bcBarrelCollection = cms.InputTag('particleFlowClusterECAL')
mustacheConversions.scIslandEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
mustacheConversions.bcEndcapCollection = cms.InputTag('particleFlowClusterECAL')
mustacheConversions.conversionIOTrackProducer = cms.string('ckfInOutTracksFromMustacheConversions')
mustacheConversions.conversionOITrackProducer = cms.string('ckfOutInTracksFromMustacheConversions')
mustacheConversionSequence = cms.Sequence(mustacheConversions)
