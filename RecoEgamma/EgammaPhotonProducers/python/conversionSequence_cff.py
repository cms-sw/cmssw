import FWCore.ParameterSet.Config as cms

#
#
# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTracks_cff import *
# converted photon producer
#from RecoEgamma.EgammaTools.PhotonConversionMVAComputer_cfi import *
from RecoEgamma.EgammaPhotonProducers.conversions_cfi import *
#conversionSequence = cms.Sequence(ckfTracksFromConversions*conversions)
conversionTask = cms.Task(conversions)
conversionSequence = cms.Sequence(conversionTask)

oldegConversions = conversions.clone(
    scHybridBarrelProducer    = "correctedHybridSuperClusters",
    bcBarrelCollection        = "hybridSuperClusters:hybridBarrelBasicClusters",
    scIslandEndcapProducer    = "correctedMulti5x5SuperClustersWithPreshower",
    bcEndcapCollection        = "multi5x5SuperClusters:multi5x5EndcapBasicClusters",
    conversionIOTrackProducer = 'ckfInOutTracksFromOldEGConversions',
    conversionOITrackProducer = 'ckfOutInTracksFromOldEGConversions'
)
oldegConversionSequence = cms.Sequence(oldegConversions)
