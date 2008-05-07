import FWCore.ParameterSet.Config as cms

#
# producer for photons
# $Id: photons_cfi.py,v 1.4 2008/05/06 19:33:50 nancy Exp $
#
photons = cms.EDProducer("PhotonProducer",
    scHybridBarrelProducer = cms.InputTag('correctedHybridSuperClusters'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    minR9 = cms.double(0.93),
    usePrimaryVertex = cms.bool(True),
    risolveConversionAmbiguity = cms.bool(True),
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt'),           
    scIslandEndcapProducer = cms.InputTag('multi5x5SuperClustersWithPreshower'),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesFromCTFTracks'),
    conversionCollection = cms.string(''),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    photonCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronPixelSeeds'),
    conversionProducer = cms.string('conversions'),
    hbheInstance = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    posCalc_t0_endc = cms.double(6.3),
    minSCEt = cms.double(5.0),
    maxHOverE = cms.double(0.2),
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    posCalc_t0_barl = cms.double(7.7)
   
)


