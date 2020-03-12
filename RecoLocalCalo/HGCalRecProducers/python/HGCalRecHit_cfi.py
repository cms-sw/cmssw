import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *

dEdX = cms.PSet(
    weights = cms.vdouble(0.0,   # there is no layer zero
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
                          46.098),

    weightsNose = cms.vdouble(0.0,   # there is no layer zero                  
                              39.500245, # Mev                                 
                              39.756638,
                              39.756638,
                              39.756638,
                              39.756638,
                              66.020266,
                              92.283895,
                              92.283895)
)

dEdX_weights_v9 = cms.vdouble(0.0,      # there is no layer zero
                              8.366557, # Mev
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              10.425456,  
                              31.497849,  
                              51.205434,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              52.030486,  
                              71.265149,  
                              90.499812,  
                              90.894274,  
                              90.537470,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205,  
                              89.786205)


from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify( dEdX, weights = dEdX_weights_v9 )

dEdX_weights_v10 = cms.vdouble(0.0,      # there is no layer zero
                               8.894541,  # Mev
                               10.937907,
                               10.937907,
                               10.937907,
                               10.937907,
                               10.937907,
                               10.937907,
                               10.937907,
                               10.937907,
                               10.932882,
                               10.932882,
                               10.937907,
                               10.937907,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               10.938169,
                               32.332097,
                               51.574301,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               51.444192,
                               69.513118,
                               87.582044,
                               87.582044,
                               87.582044,
                               87.582044,
                               87.582044,
                               87.214571,
                               86.888309,
                               86.929520,
                               86.929520,
                               86.929520)


from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify( dEdX, weights = dEdX_weights_v10 )

# HGCAL rechit producer
HGCalRecHit = cms.EDProducer(
    "HGCalRecHitProducer",
    HGCEErechitCollection = cms.string('HGCEERecHits'),
    HGCEEuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCEEUncalibRecHits'),
    HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
    HGCHEFuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCHEFUncalibRecHits'),
    HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
    HGCHEBuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCHEBUncalibRecHits'),
    HGCHFNoserechitCollection = cms.string('HGCHFNoseRecHits'),
    HGCHFNoseuncalibRecHitCollection = cms.InputTag('HGCalUncalibRecHit:HGCHFNoseUncalibRecHits'), 


    # digi constants
    HGCEE_keV2DIGI  = hgceeDigitizer.digiCfg.keV2fC,
    HGCEE_isSiFE    = HGCalUncalibRecHit.HGCEEConfig.isSiFE,
    HGCEE_fCPerMIP  = HGCalUncalibRecHit.HGCEEConfig.fCPerMIP,
    HGCHEF_keV2DIGI = hgchefrontDigitizer.digiCfg.keV2fC,
    HGCHEF_isSiFE   = HGCalUncalibRecHit.HGCHEFConfig.isSiFE,
    HGCHEF_fCPerMIP = HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP,
    HGCHEB_keV2DIGI = hgchebackDigitizer.digiCfg.keV2MIP,
    HGCHEB_isSiFE   = HGCalUncalibRecHit.HGCHEBConfig.isSiFE,
    HGCHFNose_keV2DIGI = hfnoseDigitizer.digiCfg.keV2fC,
    HGCHFNose_isSiFE   = HGCalUncalibRecHit.HGCHFNoseConfig.isSiFE,
    HGCHFNose_fCPerMIP = HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP,
    # don't produce rechit if detid is a ghost one
    rangeMask = cms.uint32(4294442496),
    rangeMatch = cms.uint32(1161838592),


    # EM Scale calibrations
    layerWeights = dEdX.weights,
    layerNoseWeights = dEdX.weightsNose,

    thicknessCorrection = cms.vdouble(1.132,1.092,1.084), # 100, 200, 300 um
    thicknessNoseCorrection = cms.vdouble(1.132,1.092,1.084), # 100, 200, 300 um
    HGCEE_noise_fC = hgceeDigitizer.digiCfg.noise_fC,
    HGCEE_cce = hgceeDigitizer.digiCfg.chargeCollectionEfficiencies,
    HGCHEF_noise_fC = hgchefrontDigitizer.digiCfg.noise_fC,
    HGCHEF_cce = hgchefrontDigitizer.digiCfg.chargeCollectionEfficiencies,
    HGCHEB_noise_MIP = hgchebackDigitizer.digiCfg.noise,
    HGCHFNose_noise_fC = hfnoseDigitizer.digiCfg.noise_fC,
    HGCHFNose_cce = hfnoseDigitizer.digiCfg.chargeCollectionEfficiencies,

    # expected resolution on time for recHits - ns units
    minValSiPar = cms.double(10.),
    maxValSiPar = cms.double(1.e4),
    noiseSiPar = cms.double(5.5),
    constSiPar = cms.double(0.02),

    # algo
    algo = cms.string("HGCalRecHitWorkerSimple")

    )

phase2_hgcalV9.toModify( HGCalRecHit , thicknessCorrection = [0.759,0.760,0.773] ) #120um, 200um, 300um
phase2_hgcalV10.toModify( HGCalRecHit , thicknessCorrection = [0.781,0.775,0.769] ) #120um, 200um, 300um

phase2_hfnose.toModify( HGCalRecHit , thicknessNoseCorrection = [0.759,0.760,0.773])
