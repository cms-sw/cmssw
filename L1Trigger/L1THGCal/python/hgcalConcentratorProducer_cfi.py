import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

conc_proc = cms.PSet( ConcProcessorName  = cms.string('HGCalConcentratorProcessor'),
		        #Method = cms.string('bestChoiceSelect'),
			Method = cms.string('thresholdSelect'), 	
			NData = cms.uint32(999),
			MaxCellsInModule = cms.uint32(116),
			linLSB = cms.double(100./1024.),
			adcsaturationBH = adcSaturationBH_MIP,
			adcnBitsBH = adcNbitsBH,
			TCThreshold_fC = cms.double(0.),
			TCThresholdBH_MIP = cms.double(0.))

hgcalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    bxCollection = cms.InputTag('hgcalVFEProducer:calibratedTriggerCells'),
    Concentratorparam = conc_proc.clone()
    )
