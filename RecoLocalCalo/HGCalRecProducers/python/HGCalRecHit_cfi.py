import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *
# HGCAL rechit producer
HGCalRecHit = cms.EDProducer(
    "HGCalRecHitProducer",
    HGCEErechitCollection = cms.string('HGCEERecHits'),
    HGCEEuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCEEUncalibRecHits"),
    HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
    HGCHEFuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEFUncalibRecHits"),
    HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
    HGCHEBuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEBUncalibRecHits"),
    
    # digi constants
    HGCEE_keV2DIGI  = hgceeDigitizer.digiCfg.keV2fC,
    HGCHEF_keV2DIGI = hgchefrontDigitizer.digiCfg.keV2fC,
    HGCHEB_keV2DIGI = hgchebackDigitizer.digiCfg.keV2MIP,
    
    # algo
    algo = cms.string("HGCalRecHitWorkerSimple")
    
    )
