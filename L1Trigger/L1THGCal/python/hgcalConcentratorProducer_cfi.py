import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

EE_DR_GROUP = 7
FH_DR_GROUP = 6
BH_DR_GROUP = 12
MAX_LAYERS = 52

STC_SIZE_CUSTOM_PERLAYER_100 = ([4] + # no layer 0
                     [4]*EE_DR_GROUP + [4]*EE_DR_GROUP + [4]*EE_DR_GROUP + [4]*EE_DR_GROUP + # EM
                     [4]*FH_DR_GROUP + [4]*FH_DR_GROUP + # FH
                     [4]*BH_DR_GROUP) # BH
STC_SIZE_CUSTOM_PERLAYER_200 = ([16] + # no layer 0
                     [16]*EE_DR_GROUP + [16]*EE_DR_GROUP + [16]*EE_DR_GROUP + [16]*EE_DR_GROUP + # EM
                     [16]*FH_DR_GROUP + [16]*FH_DR_GROUP + # FH
                     [16]*BH_DR_GROUP) # BH
STC_SIZE_CUSTOM_PERLAYER_300 = STC_SIZE_CUSTOM_PERLAYER_200
STC_SIZE_CUSTOM_PERLAYER_SCIN = STC_SIZE_CUSTOM_PERLAYER_200


CTC_SIZE =  ( [2]*(MAX_LAYERS+1)*4)
STC_SIZE =  ( [4]*(MAX_LAYERS+1)+ [16]*(MAX_LAYERS+1)*3 )


threshold_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                               Method = cms.string('thresholdSelect'),
                               threshold_silicon = cms.double(2.), # MipT
                               threshold_scintillator = cms.double(2.), # MipT
                               coarsenTriggerCells = cms.bool(False),
                               fixedDataSizePerHGCROC = cms.bool(False),
                               ctcSize = cms.vuint32(CTC_SIZE),
                               )

# Column is Nlinks, Row is NWafers
# Requested size = 8(links)x8(wafers)
# Values taken from https://indico.cern.ch/event/747610/contributions/3155360/, slide 13
# For motherboards larger than 3, it is split in two
bestchoice_ndata_centralized = [
        13, 42, 75,  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        13, 40, 74, 80, 114, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        12, 39, 72, 82, 116, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        12, 26, 53, 80, 114, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        12, 25, 52, 79, 112, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 24, 51, 78, 111, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,  0,  0,  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,  0,  0,  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]


bestchoice_ndata_decentralized = [
        1, 3, 6, 9, 14, 18, 23, 27, 32, 37, 41, 46, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0,
        ]


superTCCompression_proc = cms.PSet(exponentBits = cms.uint32(4),
                                   mantissaBits = cms.uint32(5),
                                   rounding = cms.bool(True),
)

coarseTCCompression_proc = cms.PSet(exponentBits = cms.uint32(4),
                                    mantissaBits = cms.uint32(3),
                                   rounding = cms.bool(True),
)

from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import vfe_proc
best_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.string('bestChoiceSelect'),
                          NData = cms.vuint32(bestchoice_ndata_centralized),
                          coarsenTriggerCells = cms.bool(False),
                          fixedDataSizePerHGCROC = cms.bool(False),
                          coarseTCCompression = coarseTCCompression_proc.clone(),
                          superTCCalibration = vfe_proc.clone(),
                          ctcSize = cms.vuint32(CTC_SIZE),
                          )

supertc_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('superTriggerCell'),# superTriggerCell,oneBitFraction,equalShare
                             stcSize = cms.vuint32(STC_SIZE),
                             ctcSize = cms.vuint32(CTC_SIZE),
                             fixedDataSizePerHGCROC = cms.bool(False),
                             coarsenTriggerCells = cms.bool(False),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration = vfe_proc.clone(),
                             )


mixedbcstc_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.string('mixedBestChoiceSuperTriggerCell'),
                          NData = cms.vuint32(bestchoice_ndata_centralized),
                          coarsenTriggerCells = cms.bool(False),
                          fixedDataSizePerHGCROC = cms.bool(False),
                          type_energy_division = cms.string('superTriggerCell'),# superTriggerCell,oneBitFraction,equalShare
                          stcSize = cms.vuint32(STC_SIZE),
                          ctcSize = cms.vuint32(CTC_SIZE),
                          supertccompression = superTCCompression_proc.clone(),
                          coarseTCCompression = coarseTCCompression_proc.clone(),
                          superTCCalibration = vfe_proc.clone(),
                          )


coarsetc_onebitfraction_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('oneBitFraction'),
                             stcSize = cms.vuint32([4]*(MAX_LAYERS+1)+ [8]*(MAX_LAYERS+1)*3),
                             ctcSize = cms.vuint32(CTC_SIZE),
                             fixedDataSizePerHGCROC = cms.bool(True),
                             coarsenTriggerCells = cms.bool(False),
                             oneBitFractionThreshold = cms.double(0.125),
                             oneBitFractionLowValue = cms.double(0.0625),
                             oneBitFractionHighValue = cms.double(0.25),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration = vfe_proc.clone(),
                             )


coarsetc_equalshare_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             type_energy_division = cms.string('equalShare'),
                             stcSize = cms.vuint32([4]*(MAX_LAYERS+1)+ [8]*(MAX_LAYERS+1)*3),
                             ctcSize = cms.vuint32(CTC_SIZE),
                             fixedDataSizePerHGCROC = cms.bool(True),
                             coarsenTriggerCells = cms.bool(False),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration = vfe_proc.clone(),
)





from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
# V9 samples have a different defintiion of the dEdx calibrations. To account for it
# we reascale the thresholds of the FE selection
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV9.toModify(threshold_conc_proc,
                        threshold_silicon=1.35,  # MipT
                        threshold_scintillator=1.35,  # MipT
                        )


hgcalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = threshold_conc_proc.clone()
    )
