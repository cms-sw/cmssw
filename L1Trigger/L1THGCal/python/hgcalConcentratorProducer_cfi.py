import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                     Method = cms.string('thresholdSelect'),
                     NData = cms.uint32(999),
                     MaxCellsInModule = cms.uint32(288),
                     linLSB = cms.double(100./1024.),
                     adcsaturationBH = adcSaturationBH_MIP,
                     adcnBitsBH = adcNbitsBH,
                     TCThreshold_fC = cms.double(0.),
                     TCThresholdBH_MIP = cms.double(0.),
                     triggercell_threshold_silicon = cms.double(2.), # MipT
                     triggercell_threshold_scintillator = cms.double(2.) # MipT
                     )

hgcalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = conc_proc.clone()
    )
