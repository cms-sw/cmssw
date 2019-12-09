from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
from . import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

# Silicon Digitization parameters
adcSaturation_si = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbits_si = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits
tdcSaturation_si = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC
tdcNbits_si = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits
tdcOnset_si = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
# Scintillator Digitization parameters
adcSaturation_sc = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbits_sc = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits
tdcSaturation_sc = digiparam.hgchebackDigitizer.digiCfg.feCfg.tdcSaturation_fC
tdcNbits_sc = digiparam.hgchebackDigitizer.digiCfg.feCfg.tdcNbits
tdcOnset_sc = digiparam.hgchebackDigitizer.digiCfg.feCfg.tdcOnset_fC

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

# NOSE parameters
fCPerMIPnose = recoparam.HGCalUncalibRecHit.HGCHFNoseConfig.fCPerMIP
layerWeightsNose = recocalibparam.dEdX.weightsNose
thicknessCorrectionNose = recocalibparam.HGCalRecHit.thicknessNoseCorrection

# Parameters used in several places
triggerCellLsbBeforeCompression = 100./1024.
triggerCellTruncationBits = 0

vfe_proc = cms.PSet( ProcessorName = cms.string('HGCalVFEProcessorSums'),
                    # Silicon digi parameters
                     linLSB_si = cms.double(triggerCellLsbBeforeCompression),
                     adcsaturation_si = adcSaturation_si,
                     tdcnBits_si = tdcNbits_si,
                     tdcOnset_si = tdcOnset_si,
                     adcnBits_si = adcNbits_si,
                     tdcsaturation_si = tdcSaturation_si,
                    # Scintillator digi parameters
                     linLSB_sc = cms.double(float(adcSaturation_sc.value())/(2**float(adcNbits_sc.value()))),
                     adcsaturation_sc = adcSaturation_sc,
                     tdcnBits_sc = tdcNbits_sc,
                     tdcOnset_sc = tdcOnset_sc,
                     adcnBits_sc = adcNbits_sc,
                     tdcsaturation_sc = tdcSaturation_sc,
                     linnBits = cms.uint32(16),
                     siliconCellLSB_fC =  cms.double( triggerCellLsbBeforeCompression*(2**triggerCellTruncationBits) ),
                     scintillatorCellLSB_MIP = cms.double(float(adcSaturation_sc.value())/(2**float(adcNbits_sc.value()))),
                     noiseSilicon = cms.PSet(),
                     noiseScintillator = cms.PSet(),
                     # cell thresholds before TC sums
                     # Cut at 3sigma of the noise
                     noiseThreshold = cms.double(3), # in units of sigmas of the noise
                     # Floating point compression
                     exponentBits = cms.uint32(4),
                     mantissaBits = cms.uint32(4),
                     rounding = cms.bool(True),
                     # Trigger cell calibration
                     fCperMIP = cms.double(fCPerMIP_200),
                     fCperMIPnose = cms.vdouble(fCPerMIPnose),
                     dEdXweights = layerWeights,
                     dEdXweightsNose = layerWeightsNose,
                     ThicknessCorrections = cms.vdouble(frontend_thickness_corrections),
                     thickCorr = cms.double(thicknessCorrection_200),
                     thickCorrNose = cms.vdouble(thicknessCorrectionNose),
                     )

# isolate these refs in case they aren't available in some other WF
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(vfe_proc,
    noiseSilicon = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_fC")),
    noiseScintillator = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_heback")),
)

hgcalVFEProducer = cms.EDProducer(
        "HGCalVFEProducer",
        eeDigis = cms.InputTag('simHGCalUnsuppressedDigis:EE'),
        fhDigis = cms.InputTag('simHGCalUnsuppressedDigis:HEfront'),
        bhDigis = cms.InputTag('simHGCalUnsuppressedDigis:HEback'),
        noseDigis = cms.InputTag('simHFNoseUnsuppressedDigis:HFNose'),
        ProcessorParameters = vfe_proc.clone()
       )

