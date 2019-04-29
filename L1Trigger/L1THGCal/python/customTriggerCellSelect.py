import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

def custom_triggercellselect_supertriggercell(process, 
                                              stcSize = cms.vuint32(4,4,4)
                                              ):
    
    parameters = process.hgcalConcentratorProducer.ProcessorParameters
    parameters.Method = cms.string('superTriggerCellSelect')
    parameters.stcSize = stcSize

    return process


def custom_triggercellselect_threshold(process,
        threshold=2. # in mipT
        ):
    adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
    adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

    parameters = process.hgcalConcentratorProducer.ProcessorParameters
    parameters.Method = cms.string('thresholdSelect')
    parameters.linLSB = cms.double(100./1024.)
    parameters.adcsaturationBH = adcSaturationBH_MIP
    parameters.adcnBitsBH = adcNbitsBH
    parameters.TCThreshold_fC = cms.double(0.)
    parameters.TCThresholdBH_MIP = cms.double(0.)
    parameters.triggercell_threshold_silicon = cms.double(threshold) # MipT
    parameters.triggercell_threshold_scintillator = cms.double(threshold) # MipT
    return process


def custom_triggercellselect_bestchoice(process,
       triggercells=12
        ):
    adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
    adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

    parameters = process.hgcalConcentratorProducer.ProcessorParameters
    parameters.Method = cms.string('bestChoiceSelect')
    parameters.NData = cms.uint32(triggercells)
    parameters.linLSB = cms.double(100./1024.)
    parameters.adcsaturationBH = adcSaturationBH_MIP
    parameters.adcnBitsBH = adcNbitsBH
    return process
