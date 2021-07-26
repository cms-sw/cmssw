import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photonCore collection
#
photonCore = cms.EDProducer("PhotonCoreProducer",
    conversionProducer = cms.InputTag(""),
   # conversionCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    photonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.InputTag('electronMergedSeeds'),
    minSCEt = cms.double(10.0),
    risolveConversionAmbiguity = cms.bool(True),
    endcapOnly = cms.bool(False),
#    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt')
)

photonCoreHGC = photonCore.clone(
    scHybridBarrelProducer = "",
    scIslandEndcapProducer = 'particleFlowSuperClusterHGCal',
    pixelSeedProducer = 'electronMergedSeeds',
    endcapOnly = True,
)

islandPhotonCore = photonCore.clone(
    scHybridBarrelProducer = "correctedIslandBarrelSuperClusters",
    scIslandEndcapProducer = "correctedIslandEndcapSuperClusters",
    minSCEt = 8.0
)
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(photonCore,minSCEt=0) #
egamma_lowPt_exclusive.toModify(islandPhotonCore,minSCEt = 1.0) #default 8
