import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer, hgchefrontDigitizer, hgchebackDigitizer

# HGCAL producer of rechits starting from digis
HGCalUncalibRecHit = cms.EDProducer(
    "HGCalUncalibRecHitProducer",
    HGCEEdigiCollection = cms.InputTag("mix","HGCDigisEE"),
    HGCEEhitCollection = cms.string('HGCEEUncalibRecHits'),
    HGCHEFdigiCollection = cms.InputTag("mix","HGCDigisHEfront"),
    HGCHEFhitCollection = cms.string('HGCHEFUncalibRecHits'),
    HGCHEBdigiCollection = cms.InputTag("mix","HGCDigisHEback"),
    HGCHEBhitCollection = cms.string('HGCHEBUncalibRecHits'),
    
    HGCEEConfig = cms.PSet(
        isSiFE = cms.bool(True),
        # adc information
        adcNbits      = hgceeDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits      = hgceeDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation = hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        tdcOnset      = hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC,
        toaLSB_ns     = hgceeDigitizer.digiCfg.feCfg.toaLSB_ns
        ),
    
    HGCHEFConfig = cms.PSet(
        isSiFE = cms.bool(True),
        # adc information
        adcNbits      = hgchefrontDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hgchefrontDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits      = hgchefrontDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation = hgchefrontDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        tdcOnset      = hgchefrontDigitizer.digiCfg.feCfg.tdcOnset_fC,
        toaLSB_ns     = hgchefrontDigitizer.digiCfg.feCfg.toaLSB_ns
        ),

    HGCHEBConfig = cms.PSet(
        isSiFE  = cms.bool(False),
        adcNbits      = hgchebackDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
        ),

    algo = cms.string("HGCalUncalibRecHitWorkerWeights")
)
