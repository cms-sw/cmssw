import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

def custom_triggercellselect_supertriggercell(process):
    
    process.hgcalConcentratorProducer.ProcessorParameters.Method = cms.string('superTriggerCellSelect')
    return process


def custom_triggercellselect_threshold(process):

    adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
    adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

    parameters = process.hgcalConcentratorProducer.ProcessorParameters
    parameters.Method = cms.string('thresholdSelect')
    parameters.NData = cms.uint32(999)
    parameters.MaxCellsInModule = cms.uint32(288)
    parameters.linLSB = cms.double(100./1024.)
    parameters.adcsaturationBH = adcSaturationBH_MIP
    parameters.adcnBitsBH = adcNbitsBH
    parameters.TCThreshold_fC = cms.double(0.)
    parameters.TCThresholdBH_MIP = cms.double(0.)
    parameters.triggercell_threshold_silicon = cms.double(2.) # MipT
    parameters.triggercell_threshold_scintillator = cms.double(2.) # MipT

    return process
