import FWCore.ParameterSet.Config as cms

#
# producer for photons
# $Id: photons_cfi.py,v 1.3 2008/05/06 13:23:17 hegner Exp $
#
photons = cms.EDProducer("PhotonProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    endcapHitProducer = cms.string('ecalRecHit'),
    minR9 = cms.double(0.93),
    usePrimaryVertex = cms.bool(True),
    risolveConversionAmbiguity = cms.bool(True),
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt'),           
    scIslandEndcapProducer = cms.string('multi5x5SuperClustersWithPreshower'),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesFromCTFTracks'),
    conversionCollection = cms.string(''),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    scIslandEndcapCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    photonCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronPixelSeeds'),
    conversionProducer = cms.string('conversions'),
    hbheInstance = cms.string(''),
    scHybridBarrelCollection = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    posCalc_t0_endc = cms.double(6.3),
    minSCEt = cms.double(5.0),
    maxHOverE = cms.double(0.2),
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    barrelHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_barl = cms.double(7.7)
   
)


