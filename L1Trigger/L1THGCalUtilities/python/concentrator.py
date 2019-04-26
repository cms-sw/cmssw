import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

def create_supertriggercell(process, inputs,
        stcSize = cms.vuint32(4,4,4) 
        ):
    producer = process.hgcalConcentratorProducer.clone()     
    producer.ProcessorParameters.Method = cms.string('superTriggerCellSelect')
    producer.ProcessorParameters.stcSize = stcSize
    producer.InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    producer.InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    return producer


def create_threshold(process, inputs,
        threshold=2. # in mipT
        ):
    adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
    adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits
    producer = process.hgcalConcentratorProducer.clone()
    producer.ProcessorParameters.Method = cms.string('thresholdSelect')
    producer.ProcessorParameters.linLSB = cms.double(100./1024.)
    producer.ProcessorParameters.adcsaturationBH = adcSaturationBH_MIP
    producer.ProcessorParameters.adcnBitsBH = adcNbitsBH
    producer.ProcessorParameters.TCThreshold_fC = cms.double(0.)
    producer.ProcessorParameters.TCThresholdBH_MIP = cms.double(0.)
    producer.ProcessorParameters.triggercell_threshold_silicon = cms.double(threshold) # MipT
    producer.ProcessorParameters.triggercell_threshold_scintillator = cms.double(threshold) # MipT
    producer.InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    producer.InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    return producer


def create_bestchoice(process, inputs,
       triggercells=12
        ):
    adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
    adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits
    producer = process.hgcalConcentratorProducer.clone() 
    producer.ProcessorParameters.Method = cms.string('bestChoiceSelect')
    producer.ProcessorParameters.NData = cms.uint32(triggercells)
    producer.ProcessorParameters.linLSB = cms.double(100./1024.)
    producer.ProcessorParameters.adcsaturationBH = adcSaturationBH_MIP
    producer.ProcessorParameters.adcnBitsBH = adcNbitsBH
    producer.InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    producer.InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
    return producer
