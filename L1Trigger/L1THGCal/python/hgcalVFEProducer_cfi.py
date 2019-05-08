from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
from . import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

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

vfe_proc = cms.PSet( ProcessorName = cms.string('HGCalVFEProcessorSums'),
                     linLSB = cms.double(triggerCellLsbBeforeCompression),
                     adcsaturation = adcSaturation_fC,
                     tdcnBits = tdcNbits,
                     tdcOnsetfC = tdcOnset_fC,
                     adcnBits = adcNbits,
                     tdcsaturation = tdcSaturation_fC,
                     linnBits = cms.uint32(16),
                     siliconCellLSB_fC =  cms.double( triggerCellLsbBeforeCompression*(2**triggerCellTruncationBits) ),
                     scintillatorCellLSB_MIP = cms.double(float(adcSaturationBH_MIP.value())/(2**float(adcNbitsBH.value()))),
                     # cell thresholds before TC sums
                     # Cut at 3sigma of the noise
                     thresholdsSilicon = cms.vdouble([3.*x for x in digiparam.HGCAL_noise_fC.values.value()]),
                     thresholdScintillator = cms.double(3.*digiparam.HGCAL_noise_MIP.value.value()),
                     # Floating point compression
                     exponentBits = cms.uint32(4),
                     mantissaBits = cms.uint32(4),
                     rounding = cms.bool(True),
                     # Trigger cell calibration
                     fCperMIP = cms.double(fCPerMIP_200),
                     dEdXweights = layerWeights,
                     ThicknessCorrections = cms.vdouble(frontend_thickness_corrections),
                     thickCorr = cms.double(thicknessCorrection_200)
                     )

hgcalVFEProducer = cms.EDProducer(
        "HGCalVFEProducer",
        eeDigis = cms.InputTag('simHGCalUnsuppressedDigis:EE'),
        fhDigis = cms.InputTag('simHGCalUnsuppressedDigis:HEfront'),
        bhDigis = cms.InputTag('simHGCalUnsuppressedDigis:HEback'),
        ProcessorParameters = vfe_proc.clone()
       )

