import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
conversions = cms.EDProducer("ConvertedPhotonProducer",
    #    string  bcProducer  =   "islandBasicClusters"
    #    string  bcBarrelCollection = "islandBarrelBasicClusters"
    #    string  bcEndcapCollection = "islandEndcapBasicClusters"
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    scHybridBarrelCollection = cms.string(''),
    #    string scIslandEndcapProducer   =     "correctedEndcapSuperClustersWithPreshower"
    #    string scIslandEndcapCollection =     ""
    convertedPhotonCollection = cms.string(''),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackCollection = cms.string(''),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    inOutTrackCollection = cms.string(''),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    #   string  bcProducer  =   "multi5x5BasicClusters"
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    scIslandEndcapProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    scIslandEndcapCollection = cms.string(''),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions')
)


