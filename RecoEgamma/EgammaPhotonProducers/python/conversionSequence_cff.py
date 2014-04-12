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

oldegConversions = conversions.clone()
oldegConversions.scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters")
oldegConversions.bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
oldegConversions.scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
oldegConversions.bcEndcapCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")
oldegConversions.conversionIOTrackProducer = cms.string('ckfInOutTracksFromOldEGConversions')
oldegConversions.conversionOITrackProducer = cms.string('ckfOutInTracksFromOldEGConversions')
oldegConversionSequence = cms.Sequence(oldegConversions)
