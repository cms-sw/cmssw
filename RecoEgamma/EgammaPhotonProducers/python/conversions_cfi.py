import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.egammaHBHERecHitThreshold_cff import egammaHBHERecHit
#from RecoEgamma.EgammaTools.PhotonConversionMVAComputer_cfi import *
#
#  configuration for producer of converted photons
#  
#
conversions = cms.EDProducer("ConvertedPhotonProducer",
    scHybridBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel'),
    scIslandEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower'),
    bcEndcapCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALEndcap'),
    bcBarrelCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALBarrel'),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    outInTrackCollection = cms.string(''),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions'),
    inOutTrackCollection = cms.string(''),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    convertedPhotonCollection = cms.string('uncleanedConversions'),
    generalTracksSrc = cms.InputTag("generalTracks"),
    cleanedConvertedPhotonCollection = cms.string(''),
    AlgorithmName = cms.string('ecalSeeded'),
    minSCEt = cms.double(20.0),
    hOverEConeSize = cms.double(0.15),
    hbheRecHits = egammaHBHERecHit.hbheRecHits,
    recHitEThresholdHB = egammaHBHERecHit.recHitEThresholdHB,
    recHitEThresholdHE = egammaHBHERecHit.recHitEThresholdHE,
    maxHcalRecHitSeverity = egammaHBHERecHit.maxHcalRecHitSeverity,
    maxHOverE = cms.double(0.15),
    recoverOneTrackCase = cms.bool(True),
    dRForConversionRecovery = cms.double(0.3),
    deltaCotCut = cms.double(0.05),
    minApproachDisCut  = cms.double(0.),
    maxNumOfCandidates = cms.int32(3),
    risolveConversionAmbiguity = cms.bool(True),
    maxDelta = cms.double(0.01),#delta of parameters
    maxReducedChiSq = cms.double(225.),#maximum chi^2 per degree of freedom before fit is terminated
    minChiSqImprovement = cms.double(50.),#threshold for "significant improvement" in the fit termination logic
    maxNbrOfIterations = cms.int32(40), #maximum number of convergence iterations
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt')
#    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.xml')

 )

