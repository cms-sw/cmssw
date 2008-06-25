import FWCore.ParameterSet.Config as cms

#
# producer for photons
# $Id: photons_cfi.py,v 1.12 2008/06/20 10:28:52 nancy Exp $
#
photons = cms.EDProducer("PhotonProducer",
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    minR9 = cms.double(0.93),
    usePrimaryVertex = cms.bool(True),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),
    conversionCollection = cms.string(''),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    photonCollection = cms.string(''),
    conversionProducer = cms.string('conversions'),
    risolveConversionAmbiguity = cms.bool(True),
    pixelSeedProducer = cms.string('electronPixelSeeds'),
    hbheInstance = cms.string(''),
    posCalc_t0_endc = cms.double(6.3),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    hbheModule = cms.string('hbhereco'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    minSCEt = cms.double(10.0),
    maxHOverE = cms.double(999.),
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt'),
    posCalc_t0_barl = cms.double(7.7),
    minEtRecHitBarrel = cms.double(0.),
    ecalIsolInnRBarrel = cms.double(0.045),
    ecalIsolExtRBarrel = cms.double(0.4),
    ecalIsolEtaStripBarrel = cms.double(0.02),
#    isolEtCutBarrel = cms.double(12.),
    isolEtCutBarrel = cms.double(99999.),
    minEtRecHitEndcap = cms.double(0.),
    ecalIsolInnREndcap = cms.double(0.07),
    ecalIsolExtREndcap = cms.double(0.4),
    ecalIsolEtaStripEndcap = cms.double(0.02),
#    isolEtCutEndcap = cms.double(9.)
    isolEtCutEndcap = cms.double(99999.)
                          
   
)


