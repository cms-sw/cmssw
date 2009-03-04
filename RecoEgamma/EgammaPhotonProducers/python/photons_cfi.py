import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photons
# $Id: photons_cfi.py,v 1.21 2009/02/06 18:55:07 nancy Exp $
#
photons = cms.EDProducer("PhotonProducer",
    isolationSumsCalculatorSet = cms.PSet(isolationSumsCalculator),
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),

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
    pixelSeedProducer = cms.string('ecalDrivenElectronSeeds'),
    hbheInstance = cms.string(''),
    posCalc_t0_endc = cms.double(6.3),
    # Old endcap clustering
    #    string scIslandEndcapProducer   =     "correctedEndcapSuperClustersWithPreshower"
    #    string scIslandEndcapCollection =     ""
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    hbheModule = cms.string('hbhereco'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hcalTowers = cms.InputTag("towerMaker"),
    minSCEt = cms.double(10.0),
    highEt  = cms.double(100.),                       
    minR9Barrel = cms.double(0.93),
    minR9Endcap = cms.double(0.93),                     
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    MVA_weights_location = cms.string('RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt'),
    posCalc_t0_barl = cms.double(7.7),

    maxHoverEBarrel = cms.double(0.5),
    maxHoverEEndcap = cms.double(0.5),
    ecalRecHitSumBarrel = cms.double(10.0),
    ecalRecHitSumEndcap = cms.double(10.0),
    hcalTowerSumBarrel = cms.double(5.0),
    hcalTowerSumEndcap = cms.double(10.0),
    nTrackSolidConeBarrel =cms.double(999.),
    nTrackSolidConeEndcap =cms.double(999.),
    nTrackHollowConeBarrel =cms.double(999.),
    nTrackHollowConeEndcap =cms.double(999.),
    trackPtSumSolidConeBarrel =cms.double(999.),
    trackPtSumSolidConeEndcap =cms.double(999.),                     
    trackPtSumHollowConeBarrel =cms.double(999.),
    trackPtSumHollowConeEndcap =cms.double(999.),
    sigmaIetaIetaCutBarrel=cms.double(999.),
    sigmaIetaIetaCutEndcap=cms.double(999.)
                         

)


