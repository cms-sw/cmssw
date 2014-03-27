import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.pfIsolationCalculator_cfi import *
from RecoEgamma.PhotonIdentification.isolationCalculator_cfi import *
from RecoEgamma.PhotonIdentification.mipVariable_cfi import *
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

#
# producer for photons
#
gedPhotons = cms.EDProducer("GEDPhotonProducer",
    photonProducer = cms.InputTag("gedPhotonsTmp"),                        
    reconstructionStep = cms.string("tmp"),
    #old regression <<<<<< >>>>>> do not use
    regressionWeightsFromDB =   cms.bool(True),                    
    energyRegressionWeightsFileLocation = cms.string('/afs/cern.ch/user/b/bendavid/cmspublic/regweights/gbrph.root'),
    energyRegressionWeightsDBLocation = cms.string('wgbrph'),
    # refined SC regression setup
    useRegression = cms.bool(True),
    regressionConfig = cms.PSet(
       regressionKeyEB = cms.string('gedphoton_EBCorrection_offline_v1'),
       regressionKeyEE = cms.string('gedphoton_EECorrection_offline_v1'),
       uncertaintyKeyEB = cms.string('gedphoton_EBUncertainty_offline_v1'),
       uncertaintyKeyEE = cms.string('gedphoton_EEUncertainty_offline_v1'),
       vertexCollection = cms.InputTag("offlinePrimaryVertices"),
       ecalRecHitsEB = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
       ecalRecHitsEE = cms.InputTag('ecalRecHit','EcalRecHitsEE')
       ),
    superClusterEnergyCorrFunction =  cms.string("EcalClusterEnergyCorrection"),                  
    superClusterEnergyErrorFunction = cms.string("EcalClusterEnergyUncertainty"),
    superClusterCrackEnergyCorrFunction =  cms.string("EcalClusterCrackCorrection"),                                       
    photonEcalEnergyCorrFunction = cms.string("EcalClusterEnergyCorrectionObjectSpecific"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    pfCandidates = cms.InputTag("particleFlowTmp"),                        
    outputPhotonCollection = cms.string(""),                         
    valueMapPhotons = cms.string("valMapPFEgammaCandToPhoton"),             
    #candidateP4type = cms.string("fromRegression"),
    candidateP4type = cms.string("fromRefinedSCRegression"),
    isolationSumsCalculatorSet = cms.PSet(isolationSumsCalculator),
    PFIsolationCalculatorSet = cms.PSet(pfIsolationCalculator),                        
    mipVariableSet = cms.PSet(mipVariable), 
    usePrimaryVertex = cms.bool(True),
    primaryVertexProducer = cms.InputTag('offlinePrimaryVerticesWithBS'),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    hbheInstance = cms.string(''),
    posCalc_t0_endc = cms.double(6.3),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    hbheModule = cms.string('hbhereco'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hcalTowers = cms.InputTag("towerMaker"),
    runMIPTagger = cms.bool(True),
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
    ecalRecHitSumEtOffsetBarrel = cms.double(999999999),
    ecalRecHitSumEtSlopeBarrel = cms.double(0.),
    ecalRecHitSumEtOffsetEndcap = cms.double(999999999),
    ecalRecHitSumEtSlopeEndcap = cms.double(0.),
    hcalTowerSumEtOffsetBarrel = cms.double(999999999),
    hcalTowerSumEtSlopeBarrel = cms.double(0.),
    hcalTowerSumEtOffsetEndcap = cms.double(999999999),
    hcalTowerSumEtSlopeEndcap = cms.double(0.),                      
    nTrackSolidConeBarrel =cms.double(999999999),
    nTrackSolidConeEndcap =cms.double(999999999),
    nTrackHollowConeBarrel =cms.double(999999999),
    nTrackHollowConeEndcap =cms.double(999999999),
    trackPtSumSolidConeBarrel =cms.double(999999999),
    trackPtSumSolidConeEndcap =cms.double(999999999),
    trackPtSumHollowConeBarrel =cms.double(999999999),
    trackPtSumHollowConeEndcap =cms.double(999999999),
    sigmaIetaIetaCutBarrel=cms.double(999999999),
    sigmaIetaIetaCutEndcap=cms.double(999999999),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(6.3),        
                                  T0_endcPresh = cms.double(3.6),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),                  

    RecHitFlagToBeExcludedEB = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEB = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    RecHitFlagToBeExcludedEE = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEE = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
)


