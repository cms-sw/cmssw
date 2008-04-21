import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi import *
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
import copy
from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import *
ecalUncalibHit2 = copy.deepcopy(ecalFixedAlphaBetaFitUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalUncalibHit = copy.deepcopy(ecalWeightUncalibRecHit)
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff import *
ecalConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    siteLocalConfig = cms.untracked.bool(False),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants_CosmGain200')
    ), 
        cms.PSet(
            record = cms.string('EcalPedestalsRcd'),
            tag = cms.string('EcalPedestals_online')
        ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_CosmGain200')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAlphasRcd'),
            tag = cms.string('EcalLaserAlphas_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRcd'),
            tag = cms.string('EcalLaserAPDPNRatios_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
            tag = cms.string('EcalLaserAPDPNRatiosRef_trivial')
        )),
    messagelevel = cms.untracked.uint32(0),
    timetype = cms.string('runnumber'),
    #        string connect = "frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_18X_ECAL"
    connect = cms.string('frontier://FrontierOn/CMS_COND_ON_18X_ECAL'),
    authenticationMethod = cms.untracked.uint32(1)
)

ecalBarrelDataSequence = cms.Sequence(ecalEBunpacker*ecalUncalibHit*ecalUncalibHit2*ecalRecHit*ecalTriggerPrimitiveDigis)
ecalUncalibHit2.MinAmplBarrel = 12.
ecalUncalibHit2.MinAmplEndcap = 16.
ecalUncalibHit2.EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
ecalUncalibHit2.EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
ecalUncalibHit.EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
ecalUncalibHit.EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB")
ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE")
ecalTriggerPrimitiveDigis.Label = 'ecalEBunpacker'
ecalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
ecalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'
ecalTriggerPrimitiveDigis.BarrelOnly = True
EcalTrigPrimESProducer.DatabaseFileEB = 'TPG_EB_MIPs.txt'
EcalTrigPrimESProducer.DatabaseFileEE = 'TPG_EE.txt'
islandBasicClusters.IslandBarrelSeedThr = 0.150 ## 0.500

islandBasicClusters.IslandEndcapSeedThr = 0.150 ## 0.180

hybridSuperClusters.HybridBarrelSeedThr = 0.150 ## 1.000

hybridSuperClusters.step = 1 ## 17

hybridSuperClusters.eseed = 0.150 ## 0.350

islandSuperClusters.seedTransverseEnergyThreshold = 0.150 ## 1.000


