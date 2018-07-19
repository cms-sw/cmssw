import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

# Digitization parameters
adcSaturation_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbits = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits
tdcSaturation_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC
tdcNbits = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits
tdcOnset_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

# Reco calibration parameters
fCPerMIPee = recoparam.HGCalUncalibRecHit.HGCEEConfig.fCPerMIP
fCPerMIPfh = recoparam.HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP
layerWeights = layercalibparam.TrgLayer_dEdX_weights
thicknessCorrection = recocalibparam.HGCalRecHit.thicknessCorrection

# Equalization in the frontend of the sensor responses to 200um sensors
frontend_thickness_corrections = [1./(c1*c2) for c1,c2 in zip(fCPerMIPee,thicknessCorrection)]
c200 = frontend_thickness_corrections[1]
frontend_thickness_corrections = [c/c200 for c in frontend_thickness_corrections]
fCPerMIP_200 = fCPerMIPee[1]
thicknessCorrection_200 = thicknessCorrection[1]

# Parameters used in several places
triggerCellLsbBeforeCompression = 100./1024.
triggerCellTruncationBits = 0

calib_parValues = cms.PSet( siliconCellLSB_fC =  cms.double( triggerCellLsbBeforeCompression*(2**triggerCellTruncationBits) ),
                            scintillatorCellLSB_MIP = cms.double(float(adcSaturationBH_MIP.value())/(2**float(adcNbitsBH.value()))),
                            fCperMIP = cms.double(fCPerMIP_200),
                            dEdXweights = layerWeights,
                            thickCorr = cms.double(thicknessCorrection_200)
                            )

vfe_proc = cms.PSet( ProcessorName = cms.string('HGCalVFEProcessorSums'),
                     calib_parameters = calib_parValues.clone(),
                     linLSB = cms.double(100./1024.),
                     adcsaturation = adcSaturation_fC,
                     tdcnBits = tdcNbits,
                     tdcOnsetfC = tdcOnset_fC,
                     adcnBits = adcNbits,
                     tdcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
                     linnBits = cms.uint32(16),
                     ThicknessCorrections = cms.vdouble(frontend_thickness_corrections)
                   )

hgcalVFEProducer = cms.EDProducer(
        "HGCalVFEProducer",
        eeDigis = cms.InputTag('mix:HGCDigisEE'),
        fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
        bhDigis = cms.InputTag('mix:HGCDigisHEback'),
        ProcessorParameters = vfe_proc.clone()
       )

