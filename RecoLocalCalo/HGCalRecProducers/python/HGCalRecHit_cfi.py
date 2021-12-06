import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *

dEdX = cms.PSet(
	# for v10 geometry
    weights = cms.vdouble(0.0,      # there is no layer zero
                          8.894541,  # MeV
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
                          86.929520),

    weightsNose = cms.vdouble(0.0,   # there is no layer zero
                              39.500245, # MeV
                              39.756638,
                              39.756638,
                              39.756638,
                              39.756638,
                              66.020266,
                              92.283895,
                              92.283895)
)

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

    #With the new regional em factors there are 7 different factors used. 
    #Six of them are for silicon and one for scint. For silicon it is in the following order
    # CE_E_120um, CE_E_200um, CE_E_300um, CE_H_120um, CE_H_200um, CE_H_300um
    thicknessCorrection = cms.vdouble(1.132,1.092,1.084,1.0,1.0,1.0),
    deltasi_index_regemfac = cms.int32(3),
    #One factor for scint 
    sciThicknessCorrection = cms.double(1.0),
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

# For silicon the order is: CE_E_120um, CE_E_200um, CE_E_300um, CE_H_120um, CE_H_200um, CE_H_300um
phase2_hgcalV10.toModify( HGCalRecHit , thicknessCorrection = [0.77, 0.77, 0.77, 0.84, 0.84, 0.84] , sciThicknessCorrection =  0.90 ) 

phase2_hfnose.toModify( HGCalRecHit , thicknessNoseCorrection = [0.58,0.58,0.58])
