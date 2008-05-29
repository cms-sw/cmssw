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
    bcBarrelCollection = cms.InputTag("hybridSuperClusters"),
    bcEndcapCollection  = cms.InputTag("multi5x5BasicClusters:multi5x5EndcapBasicClusters"),

#    bcEndcapCollection = cms.string('multi5x5EndcapBasicClusters'),
#    bcBarrelCollection = cms.string('multi5x5BarrelBasicClusters'),
    scIslandEndcapProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower'),
#    bcProducer = cms.string('multi5x5BasicClusters'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    scIslandEndcapCollection = cms.string(''),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions')
)


