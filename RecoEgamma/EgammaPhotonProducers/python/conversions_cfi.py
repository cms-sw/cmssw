import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
conversions = cms.EDProducer("ConvertedPhotonProducer",
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
   # scHybridBarrelCollection = cms.string(''),
    convertedPhotonCollection = cms.string(''),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackCollection = cms.string(''),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    inOutTrackCollection = cms.string(''),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
  #  scIslandEndcapCollection = cms.string(''),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions'),
    AlgorithmName = cms.string('ecalSeeded')                         
)


