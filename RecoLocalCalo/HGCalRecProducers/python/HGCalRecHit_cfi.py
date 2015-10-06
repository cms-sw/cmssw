import FWCore.ParameterSet.Config as cms

# from SimCalorimetry.HGCSimProducers.hgcDigiProducers_cff import *
#from SimGeneral.MixingModule.hgcalDigitizer_cfi import *
# HGCAL rechit producer
HGCalRecHit = cms.EDProducer("HGCalRecHitProducer",
                             HGCEErechitCollection = cms.string('HGCEERecHits'),
                             HGCEEuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCEEUncalibRecHits"),
                             HGCHEFrechitCollection = cms.string('HGCHEFRecHits'),
                             HGCHEFuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEFUncalibRecHits"),
                             HGCHEBrechitCollection = cms.string('HGCHEBRecHits'),
                             HGCHEBuncalibRecHitCollection = cms.InputTag("HGCalUncalibRecHit","HGCHEBUncalibRecHits"),
                             # digi constants
                             
                             HGCEEmipInKeV = cms.double(55.1),#hgceeDigitizer.digiCfg.mipInKeV,
                             HGCEElsbInMIP = cms.double(0.25),#hgceeDigitizer.digiCfg.lsbInMIP,
                             HGCEEmip2noise = cms.double(7.0),#cms.double(hgceeDigitizer.digiCfg.mip2noise,

                             HGCHEFmipInKeV = cms.double(85.0),#hgchefrontDigitizer.digiCfg.mipInKeV,
                             HGCHEFlsbInMIP = cms.double(0.25),#hgchefrontDigitizer.digiCfg.lsbInMIP,
                             HGCHEFmip2noise = cms.double(7.0),#hgchefrontDigitizer.digiCfg.mip2noise,

                             HGCHEBmipInKeV = cms.double(1498.4),#hgchebackDigitizer.digiCfg.mipInKeV,
                             HGCHEBlsbInMIP = cms.double(0.25),#hgchebackDigitizer.digiCfg.lsbInMIP,
                             HGCHEBmip2noise = cms.double(5.0),#hgchebackDigitizer.digiCfg.mip2noise,
                             
                             # algo
                             algo = cms.string("HGCalRecHitWorkerSimple")
                             
                             )
