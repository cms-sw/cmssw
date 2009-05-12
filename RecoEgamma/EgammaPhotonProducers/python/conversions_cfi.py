import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  
#
conversions = cms.EDProducer("ConvertedPhotonProducer",
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    outInTrackCollection = cms.string(''),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions'),
    inOutTrackCollection = cms.string(''),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    convertedPhotonCollection = cms.string('uncleanedConversions'),
    hcalTowers = cms.InputTag("towerMaker"),                         
    cleanedConvertedPhotonCollection = cms.string(''),                         
    AlgorithmName = cms.string('ecalSeeded'),
    minSCEt = cms.double(10.0),
    hOverEConeSize = cms.double(0.15),
    maxHOverE = cms.double(0.2),                         
    recoverOneTrackCase = cms.bool(True),
    dRForConversionRecovery = cms.double(0.3),                      
    deltaCotCut = cms.double(0.05),
    minApproachDisCut  = cms.double(0.),
    maxNumOfCandidates = cms.int32(3),
    risolveConversionAmbiguity = cms.bool(True),                         
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt')
)

