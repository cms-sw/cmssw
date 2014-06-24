import FWCore.ParameterSet.Config as cms

# from SimCalorimetry.HGCSimProducers.hgcDigiProducers_cff import *
from SimGeneral.MixingModule.hgcalDigitizer_cfi import *
# HGCAL rechit producer
HGCalRecHit = cms.EDProducer("HGCalRecHitProducer",
                             HGCEErechitCollection = cms.string('HGCEERecHits'),
                             HGCEEuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCEEUncalibRecHits"),
                             HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
                             HGCHEFuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEFUncalibRecHits"),
                             HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
                             HGCHEBuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEBUncalibRecHits"),
                             # digi constants
                             
                             HGCEEmipInKeV = hgceeDigitizer.digiCfg.mipInKeV,
                             HGCEElsbInMIP = hgceeDigitizer.digiCfg.lsbInMIP,
                             HGCEEmip2noise = hgceeDigitizer.digiCfg.mip2noise,

                             HGCHEFmipInKeV = hgchefrontDigitizer.digiCfg.mipInKeV,
                             HGCHEFlsbInMIP = hgchefrontDigitizer.digiCfg.lsbInMIP,
                             HGCHEFmip2noise = hgchefrontDigitizer.digiCfg.mip2noise,

                             HGCHEBmipInKeV = hgchebackDigitizer.digiCfg.mipInKeV,
                             HGCHEBlsbInMIP = hgchebackDigitizer.digiCfg.lsbInMIP,
                             HGCHEBmip2noise = hgchebackDigitizer.digiCfg.mip2noise,
                             
                             # algo
                             algo = cms.string("HGCalRecHitWorkerSimple")
                             
                             )
