import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer, hgchefrontDigitizer, hgchebackDigitizer, hfnoseDigitizer

fCPerMIP_mpv = cms.vdouble(1.25,2.57,3.88) #120um, 200um, 300um
fCPerMIP_mean = cms.vdouble(2.06,3.43,5.15) #120um, 200um, 300um

# HGCAL producer of rechits starting from digis
HGCalUncalibRecHit = cms.EDProducer(
    "HGCalUncalibRecHitProducer",
    HGCEEdigiCollection = cms.InputTag('hgcalDigis:EE'),
    HGCEEhitCollection = cms.string('HGCEEUncalibRecHits'),
    HGCHEFdigiCollection = cms.InputTag('hgcalDigis:HEfront'),
    HGCHEFhitCollection = cms.string('HGCHEFUncalibRecHits'),
    HGCHEBdigiCollection = cms.InputTag('hgcalDigis:HEback'),
    HGCHEBhitCollection = cms.string('HGCHEBUncalibRecHits'),
    HGCHFNosedigiCollection = cms.InputTag('hfnoseDigis:HFNose'),
    HGCHFNosehitCollection = cms.string('HGCHFNoseUncalibRecHits'),
    
    HGCEEConfig = cms.PSet(
        isSiFE = cms.bool(True),
        # adc information
        adcNbits      = hgceeDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits      = hgceeDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation = hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        tdcOnset      = hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC,
        toaLSB_ns     = hgceeDigitizer.digiCfg.feCfg.toaLSB_ns,
        tofDelay      = hgceeDigitizer.tofDelay,
        fCPerMIP      = fCPerMIP_mpv
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
        toaLSB_ns     = hgchefrontDigitizer.digiCfg.feCfg.toaLSB_ns,
        tofDelay      = hgchefrontDigitizer.tofDelay,
        fCPerMIP      = fCPerMIP_mpv
        ),

    HGCHEBConfig = cms.PSet(
        isSiFE  = cms.bool(True),
        # adc information
        adcNbits      = hgchebackDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits      = hgchebackDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation = hgchebackDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        tdcOnset      = hgchebackDigitizer.digiCfg.feCfg.tdcOnset_fC,
        toaLSB_ns     = hgchebackDigitizer.digiCfg.feCfg.toaLSB_ns,
        tofDelay      = hgchebackDigitizer.tofDelay,
        fCPerMIP      = cms.vdouble(1.0,1.0,1.0) #dummy values, it's scintillator
        ),

    HGCHFNoseConfig = cms.PSet(
        isSiFE = cms.bool(False),
        # adc information
        adcNbits      = hfnoseDigitizer.digiCfg.feCfg.adcNbits,
        adcSaturation = hfnoseDigitizer.digiCfg.feCfg.adcSaturation_fC,
        #tdc information
        tdcNbits      = hfnoseDigitizer.digiCfg.feCfg.tdcNbits,
        tdcSaturation = hfnoseDigitizer.digiCfg.feCfg.tdcSaturation_fC,
        tdcOnset      = hfnoseDigitizer.digiCfg.feCfg.tdcOnset_fC,
        toaLSB_ns     = hfnoseDigitizer.digiCfg.feCfg.toaLSB_ns,
        tofDelay      = hfnoseDigitizer.tofDelay,
        fCPerMIP      = fCPerMIP_mpv
        ),

    algo = cms.string("HGCalUncalibRecHitWorkerWeights")
)

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify( HGCalUncalibRecHit.HGCEEConfig , fCPerMIP = fCPerMIP_mean ) 
phase2_hgcalV10.toModify( HGCalUncalibRecHit.HGCHEFConfig , fCPerMIP = fCPerMIP_mean )

from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16
phase2_hgcalV16.toModify( HGCalUncalibRecHit.HGCEEConfig , fCPerMIP = fCPerMIP_mean ) 
phase2_hgcalV16.toModify( HGCalUncalibRecHit.HGCHEFConfig , fCPerMIP = fCPerMIP_mean )

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toModify( HGCalUncalibRecHit.HGCHFNoseConfig ,
          isSiFE = True ,
          fCPerMIP = fCPerMIP_mean
)
