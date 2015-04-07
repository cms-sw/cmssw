import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.hgcalDigitizer_cfi import hgceeDigitizer, hgchefrontDigitizer, hgchebackDigitizer

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
        # number of fC in a MIP
        mipInfC = hgceeDigitizer.digiCfg.mipInfC,
        # adc information
        adcNbits = hgceeDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation_fC = hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits = hgceeDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation_fC = hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        toaLSB_ns = hgceeDigitizer.digiCfg.feCfg.toaLSB_ns
        ),
    
    HGCHEFConfig = cms.PSet(
        isSiFE = cms.bool(True),
        # number of fC in a MIP
        mipInfC = hgchefrontDigitizer.digiCfg.mipInfC,
        # adc information
        adcNbits = hgchefrontDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation_fC = hgchefrontDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits = hgchefrontDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation_fC = hgchefrontDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        toaLSB_ns = hgchefrontDigitizer.digiCfg.feCfg.toaLSB_ns
        ),

    HGCHEBConfig = cms.PSet(
        isSiFE = cms.bool(False),
        lsbInMIP = hgchebackDigitizer.digiCfg.feCfg.lsbInMIP
        ),

    algo = cms.string("HGCalUncalibRecHitWorkerWeights")
)
