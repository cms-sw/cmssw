import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits

# MAX_LAYERS should be equal to kNHGCalLayersMax_ defined in interface/HGCalCoarseTriggerCellMapping.h
# MAX_LAYERS can be larger than the actual number of layers
# CTC / STC sizes vectors should have a length of 4*MAX_LAYERS, 4 = 3 different silicon thicknesses + scintillator portion
MAX_LAYERS = 52
CTC_2_SIZES = cms.vuint32( [2]*(MAX_LAYERS+1)*4 )
STC_4_AND_16_SIZES = cms.vuint32( [4]*(MAX_LAYERS+1)+ [16]*(MAX_LAYERS+1)*3 )
STC_4_AND_8_SIZES = cms.vuint32( [4]*(MAX_LAYERS+1)+ [8]*(MAX_LAYERS+1)*3 )

threshold_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                               Method = cms.vstring(['thresholdSelect']*3),
                               threshold_silicon = cms.double(2.), # MipT
                               threshold_scintillator = cms.double(2.), # MipT
                               coarsenTriggerCells = cms.vuint32(0,0,0),
                               fixedDataSizePerHGCROC = cms.bool(False),
                               allTrigCellsInTrigSums = cms.bool(True),
                               ctcSize = CTC_2_SIZES,
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


# Values taken from ECON-T working document v9 (March 2022)
# https://edms.cern.ch/file/2206779/1/ECON-T_specification_working_doc_v9_2mar2022.pdf
bestchoice_ndata_decentralized = [
        1, 4, 6, 9, 14, 18, 23, 28, 32, 37, 41, 46, 48, 0, 0, 0,
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
                                   truncationBits = cms.uint32(0),
                                   rounding = cms.bool(True),
)

coarseTCCompression_proc = cms.PSet(exponentBits = cms.uint32(4),
                                    mantissaBits = cms.uint32(3),
                                    truncationBits = cms.uint32(0),
                                    rounding = cms.bool(True),
)

from L1Trigger.L1THGCal.l1tHGCalVFEProducer_cfi import vfe_proc
best_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.vstring(['bestChoiceSelect']*3),
                          NData = cms.vuint32(bestchoice_ndata_decentralized),
                          coarsenTriggerCells = cms.vuint32(0,0,0),
                          fixedDataSizePerHGCROC = cms.bool(False),
                          allTrigCellsInTrigSums = cms.bool(False),
                          coarseTCCompression = coarseTCCompression_proc.clone(),
                          superTCCalibration_ee = vfe_proc.calibrationCfg_ee.clone(),
                          superTCCalibration_hesi = vfe_proc.calibrationCfg_hesi.clone(),
                          superTCCalibration_hesc = vfe_proc.calibrationCfg_hesc.clone(),
                          superTCCalibration_nose = vfe_proc.calibrationCfg_nose.clone(),
                          ctcSize = CTC_2_SIZES,
                          )

supertc_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.vstring(['superTriggerCellSelect']*3),
                             type_energy_division = cms.string('superTriggerCell'),# superTriggerCell,oneBitFraction,equalShare
                             stcSize = STC_4_AND_16_SIZES,
                             ctcSize = CTC_2_SIZES,
                             fixedDataSizePerHGCROC = cms.bool(False),
                             allTrigCellsInTrigSums = cms.bool(False),
                             coarsenTriggerCells = cms.vuint32(0,0,0),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration_ee = vfe_proc.calibrationCfg_ee.clone(),
                             superTCCalibration_hesi = vfe_proc.calibrationCfg_hesi.clone(),
                             superTCCalibration_hesc = vfe_proc.calibrationCfg_hesc.clone(),
                             superTCCalibration_nose = vfe_proc.calibrationCfg_nose.clone(),
                             )

custom_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.vstring('bestChoiceSelect','superTriggerCellSelect','superTriggerCellSelect'),
                          NData = cms.vuint32(bestchoice_ndata_decentralized),
                          threshold_silicon = cms.double(2.), # MipT
                          threshold_scintillator = cms.double(2.), # MipT
                          coarsenTriggerCells = cms.vuint32(0,0,0),
                          fixedDataSizePerHGCROC = cms.bool(False),
                          allTrigCellsInTrigSums = cms.bool(False),
                          type_energy_division = cms.string('superTriggerCell'),# superTriggerCell,oneBitFraction,equalShare
                          stcSize = STC_4_AND_16_SIZES,
                          ctcSize = CTC_2_SIZES,
                          superTCCompression = superTCCompression_proc.clone(),
                          coarseTCCompression = coarseTCCompression_proc.clone(),
                          superTCCalibration_ee = vfe_proc.calibrationCfg_ee.clone(),
                          superTCCalibration_hesi = vfe_proc.calibrationCfg_hesi.clone(),
                          superTCCalibration_hesc = vfe_proc.calibrationCfg_hesc.clone(),
                          superTCCalibration_nose = vfe_proc.calibrationCfg_nose.clone(),
                          )


coarsetc_onebitfraction_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.vstring(['superTriggerCellSelect']*3),
                             type_energy_division = cms.string('oneBitFraction'),
                             stcSize = STC_4_AND_8_SIZES,
                             ctcSize = CTC_2_SIZES,
                             fixedDataSizePerHGCROC = cms.bool(True),
                             allTrigCellsInTrigSums = cms.bool(False),
                             coarsenTriggerCells = cms.vuint32(0,0,0),
                             oneBitFractionThreshold = cms.double(0.125),
                             oneBitFractionLowValue = cms.double(0.0625),
                             oneBitFractionHighValue = cms.double(0.25),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration_ee = vfe_proc.calibrationCfg_ee.clone(),
                             superTCCalibration_hesi = vfe_proc.calibrationCfg_hesi.clone(),
                             superTCCalibration_hesc = vfe_proc.calibrationCfg_hesc.clone(),
                             superTCCalibration_nose = vfe_proc.calibrationCfg_nose.clone(),
                             )


coarsetc_equalshare_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.vstring(['superTriggerCellSelect']*3),
                             type_energy_division = cms.string('equalShare'),
                             stcSize = STC_4_AND_8_SIZES,
                             ctcSize = CTC_2_SIZES,
                             fixedDataSizePerHGCROC = cms.bool(True),
                             allTrigCellsInTrigSums = cms.bool(False),
                             coarsenTriggerCells = cms.vuint32(0,0,0),
                             superTCCompression = superTCCompression_proc.clone(),
                             coarseTCCompression = coarseTCCompression_proc.clone(),
                             superTCCalibration_ee = vfe_proc.calibrationCfg_ee.clone(),
                             superTCCalibration_hesi = vfe_proc.calibrationCfg_hesi.clone(),
                             superTCCalibration_hesc = vfe_proc.calibrationCfg_hesc.clone(),
                             superTCCalibration_nose = vfe_proc.calibrationCfg_nose.clone(),
)


autoencoder_triggerCellRemap = [0,16, 32,
                                1,17, 33,
                                2,18, 34,
                                3,19, 35,
                                4,20, 36,
                                5,21, 37,
                                6,22, 38,
                                7,23, 39,
                                8,24, 40,
                                9,25, 41,
                                10,26, 42,
                                11,27, 43,
                                12,28, 44,
                                13,29, 45,
                                14,30, 46,
                                15,31, 47]

autoEncoder_bitsPerOutputLink = cms.vint32([0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])

autoEncoder_training_2eLinks = cms.PSet(encoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/encoder_2eLinks_PUdriven_constantgraph.pb'),
                                        decoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/decoder_2eLinks_PUdriven_constantgraph.pb'))

autoEncoder_training_3eLinks = cms.PSet(encoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/encoder_3eLinks_PUdriven_constantgraph.pb'),
                                        decoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/decoder_3eLinks_PUdriven_constantgraph.pb'))

autoEncoder_training_4eLinks = cms.PSet(encoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/encoder_4eLinks_PUdriven_constantgraph.pb'),
                                        decoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/decoder_4eLinks_PUdriven_constantgraph.pb'))

autoEncoder_training_5eLinks = cms.PSet(encoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/encoder_5eLinks_PUdriven_constantgraph.pb'),
                                        decoderModelFile = cms.FileInPath('L1Trigger/L1THGCal/data/decoder_5eLinks_PUdriven_constantgraph.pb'))

linkToGraphMapping = [0,0,0,1,2,3,3,3,3,3,3,3,3,3,3]

autoEncoder_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                                 Method = cms.vstring(['autoEncoder','autoEncoder','thresholdSelect']),
                                 cellRemap = cms.vint32(autoencoder_triggerCellRemap),
                                 cellRemapNoDuplicates = cms.vint32(autoencoder_triggerCellRemap),
                                 encoderShape = cms.vuint32(1,4,4,3),
                                 decoderShape = cms.vuint32(1,16),
                                 nBitsPerInput = cms.int32(8),
                                 maxBitsPerOutput = cms.int32(9),
                                 bitsPerLink = autoEncoder_bitsPerOutputLink,
                                 modelFiles = cms.VPSet([autoEncoder_training_2eLinks, autoEncoder_training_3eLinks, autoEncoder_training_4eLinks, autoEncoder_training_5eLinks]),
                                 linkToGraphMap = cms.vuint32(linkToGraphMapping),
                                 zeroSuppresionThreshold = cms.double(0.1),
                                 bitShiftNormalization = cms.bool(True),
                                 saveEncodedValues = cms.bool(False),
                                 preserveModuleSum = cms.bool(True),
                                 threshold_silicon = cms.double(2.), # MipT
                                 threshold_scintillator = cms.double(2.), # MipT
                                 type_energy_division = supertc_conc_proc.type_energy_division,
                                 stcSize = supertc_conc_proc.stcSize,
                                 ctcSize = supertc_conc_proc.ctcSize,
                                 fixedDataSizePerHGCROC = supertc_conc_proc.fixedDataSizePerHGCROC,
                                 allTrigCellsInTrigSums = supertc_conc_proc.allTrigCellsInTrigSums,
                                 coarsenTriggerCells = supertc_conc_proc.coarsenTriggerCells,
                                 superTCCompression = superTCCompression_proc.clone(),
                                 coarseTCCompression = coarseTCCompression_proc.clone(),
                                 superTCCalibration = vfe_proc.clone(),
)




from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
# >= V9 samples have a different definition of the dEdx calibrations. To account for it
# we rescale the thresholds of the FE selection
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV10.toModify(threshold_conc_proc,
                        threshold_silicon=1.35,  # MipT
                        threshold_scintillator=1.35,  # MipT
                        )


l1tHGCalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    InputTriggerCells = cms.InputTag('l1tHGCalVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('l1tHGCalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = threshold_conc_proc.clone()
    )


l1tHGCalConcentratorProducerHFNose = l1tHGCalConcentratorProducer.clone(
    InputTriggerCells = cms.InputTag('l1tHFnoseVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('l1tHFnoseVFEProducer:HGCalVFEProcessorSums')
)

