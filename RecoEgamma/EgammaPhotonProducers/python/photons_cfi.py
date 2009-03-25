import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
#
# producer for photons
# $Id: photons_cfi.py,v 1.23 2009/03/25 11:16:32 nancy Exp $
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
    minR9Barrel = cms.double(0.93),
    minR9Endcap = cms.double(0.93),                     
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    posCalc_t0_barl = cms.double(7.7),
    minSCEtBarrel = cms.double(10.0),
    minSCEtEndcap = cms.double(10.0),                     
    maxHoverEBarrel = cms.double(0.5),
    maxHoverEEndcap = cms.double(0.5),
    ecalRecHitSumEtBarrel = cms.double(10.0),
    ecalRecHitSumEtEndcap = cms.double(10.0),
    hcalTowerSumEtBarrel = cms.double(5.0),
    hcalTowerSumEtEndcap = cms.double(10.0),
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


