import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *

dEdX_weights = cms.vdouble(0.0,   # there is no layer zero
                           8.603, # Mev
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.0675, 
                           8.9515, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           10.135, 
                           11.682, 
                           13.654, 
                           13.654, 
                           13.654, 
                           13.654, 
                           13.654, 
                           13.654, 
                           13.654, 
                           38.2005, 
                           55.0265, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           49.871, 
                           62.005, 
                           83.1675, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           92.196, 
                           46.098)

# HGCAL rechit producer
HGCalRecHit = cms.EDProducer(
    "HGCalRecHitProducer",
    HGCEErechitCollection = cms.string('HGCEERecHits'),
    HGCEEuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCEEUncalibRecHits'),
    HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
    HGCHEFuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCHEFUncalibRecHits'),
    HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
    HGCHEBuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCHEBUncalibRecHits'),
    
    # digi constants
    HGCEE_keV2DIGI  = hgceeDigitizer.digiCfg.keV2fC,
    HGCEE_isSiFE    = HGCalUncalibRecHit.HGCEEConfig.isSiFE,
    HGCEE_fCPerMIP  = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP,
    HGCHEF_keV2DIGI = hgchefrontDigitizer.digiCfg.keV2fC,
    HGCHEF_isSiFE   = HGCalUncalibRecHit.HGCHEFConfig.isSiFE,
    HGCHEF_fCPerMIP = HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP,
    HGCHEB_keV2DIGI = hgchebackDigitizer.digiCfg.keV2MIP,
    HGCHEB_isSiFE   = HGCalUncalibRecHit.HGCHEBConfig.isSiFE,
    # don't produce rechit if detid is a ghost one
    rangeMask = cms.uint32(4294442496),
    rangeMatch = cms.uint32(1161838592),


    # EM Scale calibrations
    layerWeights = dEdX_weights,

    thicknessCorrection = cms.vdouble(1.132,1.092,1.084), # 100, 200, 300 um 
    HGCEE_noise_fC = hgceeDigitizer.digiCfg.noise_fC,
    HGCEE_cce = hgceeDigitizer.digiCfg.chargeCollectionEfficiency,
    HGCHEF_noise_fC = hgchefrontDigitizer.digiCfg.noise_fC,
    HGCHEF_cce = hgchefrontDigitizer.digiCfg.chargeCollectionEfficiency,
    HGCHEB_noise_MIP = hgchebackDigitizer.digiCfg.noise_MIP,
    # algo
    algo = cms.string("HGCalRecHitWorkerSimple")
    
    )


