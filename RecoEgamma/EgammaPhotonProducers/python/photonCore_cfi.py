import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photonCore collection
# $Id: photonCore_cfi.py,v 1.3 2010/02/22 19:40:26 nancy Exp $
#
photonCore = cms.EDProducer("PhotonCoreProducer",
    conversionProducer = cms.InputTag("conversions"),
   # conversionCollection = cms.string(''),
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    photonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronMergedSeeds'),
    minSCEt = cms.double(10.0),
    risolveConversionAmbiguity = cms.bool(True),
#    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt')
)


