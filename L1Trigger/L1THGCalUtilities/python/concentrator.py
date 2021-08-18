
import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc, coarsetc_onebitfraction_proc, custom_conc_proc, autoEncoder_conc_proc


def create_supertriggercell(process, inputs,
                            stcSize=supertc_conc_proc.stcSize,
                            type_energy_division=supertc_conc_proc.type_energy_division,
                            fixedDataSizePerHGCROC=supertc_conc_proc.fixedDataSizePerHGCROC,
                            coarsenTriggerCells=supertc_conc_proc.coarsenTriggerCells,
                            ctcSize=supertc_conc_proc.ctcSize,
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = supertc_conc_proc.clone(
            stcSize = stcSize,
            type_energy_division = type_energy_division,
            fixedDataSizePerHGCROC = fixedDataSizePerHGCROC,
            coarsenTriggerCells = coarsenTriggerCells,
            ctcSize = ctcSize,
            )
    return producer


def create_threshold(process, inputs,
                     threshold_silicon=threshold_conc_proc.threshold_silicon,  # in mipT
                     threshold_scintillator=threshold_conc_proc.threshold_scintillator  # in mipT
                     ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = threshold_conc_proc.clone(
            threshold_silicon = threshold_silicon,  # MipT
            threshold_scintillator = threshold_scintillator  # MipT
            )
    return producer


def create_bestchoice(process, inputs,
                      triggercells=best_conc_proc.NData,
                      coarsenTriggerCells=best_conc_proc.coarsenTriggerCells,
                      ctcSize=best_conc_proc.ctcSize,
                      ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = best_conc_proc.clone(
            NData = triggercells,
            coarsenTriggerCells = coarsenTriggerCells,
            ctcSize=ctcSize,
            )
    return producer


def create_autoencoder(process, inputs,
                       cellRemap = autoEncoder_conc_proc.cellRemap,
                       cellRemapNoDuplicates = autoEncoder_conc_proc.cellRemapNoDuplicates,
                       nBitsPerInput = autoEncoder_conc_proc.nBitsPerInput,
                       maxBitsPerOutput = autoEncoder_conc_proc.maxBitsPerOutput,
                       bitsPerLink = autoEncoder_conc_proc.bitsPerLink,
                       modelFiles = autoEncoder_conc_proc.modelFiles,
                       linkToGraphMap = autoEncoder_conc_proc.linkToGraphMap,
                       encoderShape = autoEncoder_conc_proc.encoderShape,
                       decoderShape = autoEncoder_conc_proc.decoderShape,
                       zeroSuppresionThreshold = autoEncoder_conc_proc.zeroSuppresionThreshold,
                       saveEncodedValues = autoEncoder_conc_proc.saveEncodedValues,
                       preserveModuleSum = autoEncoder_conc_proc.preserveModuleSum,
                       scintillatorMethod = 'thresholdSelect',
                     ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = autoEncoder_conc_proc.clone(
            cellRemap = cellRemap,
            cellRemapNoDuplicates = cellRemapNoDuplicates,
            nBitsPerInput = nBitsPerInput,
            maxBitsPerOutput = maxBitsPerOutput,
            bitsPerLink = bitsPerLink,
            modelFiles = modelFiles,
            linkToGraphMap = linkToGraphMap,
            encoderShape = encoderShape,
            decoderShape = decoderShape,
            zeroSuppresionThreshold = zeroSuppresionThreshold,
            saveEncodedValues = saveEncodedValues,
            preserveModuleSum = preserveModuleSum,
            Method = cms.vstring(['autoEncoder','autoEncoder', scintillatorMethod]),
            )
    return producer


def create_onebitfraction(process, inputs,
                            stcSize=coarsetc_onebitfraction_proc.stcSize,
                            fixedDataSizePerHGCROC=coarsetc_onebitfraction_proc.fixedDataSizePerHGCROC
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = coarsetc_onebitfraction_proc.clone(
            stcSize = stcSize,
            fixedDataSizePerHGCROC = fixedDataSizePerHGCROC
            )
    return producer


def create_mixedfeoptions(process, inputs,
                            stcSize=custom_conc_proc.stcSize,
                            type_energy_division=custom_conc_proc.type_energy_division,
                            fixedDataSizePerHGCROC=custom_conc_proc.fixedDataSizePerHGCROC,
                            triggercells=custom_conc_proc.NData
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = custom_conc_proc.clone(
            stcSize = stcSize,
            type_energy_division = type_energy_division,
            fixedDataSizePerHGCROC = fixedDataSizePerHGCROC,
            NData = triggercells,
            Method = cms.vstring('bestChoiceSelect','superTriggerCellSelect','superTriggerCellSelect'),        
            )
    return producer


def create_custom(process, inputs,
                            stcSize=custom_conc_proc.stcSize,
                            type_energy_division=custom_conc_proc.type_energy_division,
                            fixedDataSizePerHGCROC=custom_conc_proc.fixedDataSizePerHGCROC,
                            triggercells=custom_conc_proc.NData,
                            threshold_silicon=custom_conc_proc.threshold_silicon,  # in mipT
                            threshold_scintillator=custom_conc_proc.threshold_scintillator,  # in mipT
                            Method = custom_conc_proc.Method,
                            coarsenTriggerCells=custom_conc_proc.coarsenTriggerCells,
                            ctcSize=custom_conc_proc.ctcSize,
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = custom_conc_proc.clone(
            stcSize = stcSize,
            type_energy_division = type_energy_division,
            fixedDataSizePerHGCROC = fixedDataSizePerHGCROC,
            NData = triggercells,
            threshold_silicon = threshold_silicon,  # MipT
            threshold_scintillator = threshold_scintillator,  # MipT
            Method = Method,
            coarsenTriggerCells=coarsenTriggerCells,
            ctcSize = ctcSize,
            )
    return producer

