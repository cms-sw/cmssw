import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photonCore collection
# $Id: photonCore_cfi.py,v nancy Exp $
#
photonCore = cms.EDProducer("PhotonCoreProducer",
    conversionProducer = cms.string('conversions'),
    conversionCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    photonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.string('ecalDrivenElectronSeeds'),
    minSCEt = cms.double(10.0),
    risolveConversionAmbiguity = cms.bool(True),
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt')
)


