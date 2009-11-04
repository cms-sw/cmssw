import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photons
# $Id: photons_cfi.py,v 1.32 2009/11/03 20:52:15 nancy Exp $
#
photons = cms.EDProducer("PhotonProducer",
    photonCoreProducer = cms.string('photonCore'),
    photonCollection = cms.string(''),
    isolationSumsCalculatorSet = cms.PSet(isolationSumsCalculator),
    usePrimaryVertex = cms.bool(True),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    hbheInstance = cms.string(''),
    posCalc_t0_endc = cms.double(6.3),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    hbheModule = cms.string('hbhereco'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hcalTowers = cms.InputTag("towerMaker"),
    highEt  = cms.double(100.),                       
    minR9Barrel = cms.double(0.94),
    minR9Endcap = cms.double(0.95),                     
    hOverEConeSize = cms.double(0.15),
    posCalc_x0 = cms.double(0.89),
    posCalc_t0_barl = cms.double(7.7),
    minSCEtBarrel = cms.double(10.0),
    minSCEtEndcap = cms.double(10.0),                     
    maxHoverEBarrel = cms.double(0.5),
    maxHoverEEndcap = cms.double(0.5),
    ecalRecHitSumEtOffsetBarrel = cms.double(1.79769e+308),
    ecalRecHitSumEtSlopeBarrel = cms.double(0.),
    ecalRecHitSumEtOffsetEndcap = cms.double(1.79769e+308),
    ecalRecHitSumEtSlopeEndcap = cms.double(0.),
    hcalTowerSumEtOffsetBarrel = cms.double(1.79769e+308),
    hcalTowerSumEtSlopeBarrel = cms.double(0.),
    hcalTowerSumEtOffsetEndcap = cms.double(1.79769e+308),
    hcalTowerSumEtSlopeEndcap = cms.double(0.),                      
    nTrackSolidConeBarrel =cms.double(1.79769e+308),
    nTrackSolidConeEndcap =cms.double(1.79769e+308),
    nTrackHollowConeBarrel =cms.double(1.79769e+308),
    nTrackHollowConeEndcap =cms.double(1.79769e+308),
    trackPtSumSolidConeBarrel =cms.double(1.79769e+308),
    trackPtSumSolidConeEndcap =cms.double(1.79769e+308),                     
    trackPtSumHollowConeBarrel =cms.double(1.79769e+308),
    trackPtSumHollowConeEndcap =cms.double(1.79769e+308),
    sigmaIetaIetaCutBarrel=cms.double(1.79769e+308),
    sigmaIetaIetaCutEndcap=cms.double(1.79769e+308)
                         

)


