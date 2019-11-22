
import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc, coarsetc_onebitfraction_proc, mixedbcstc_conc_proc


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
                            stcSize=supertc_conc_proc.stcSize,
                            type_energy_division=supertc_conc_proc.type_energy_division,
                            fixedDataSizePerHGCROC=supertc_conc_proc.fixedDataSizePerHGCROC,
                            triggercells=best_conc_proc.NData
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = mixedbcstc_conc_proc.clone(
            stcSize = stcSize,
            type_energy_division = type_energy_division,
            fixedDataSizePerHGCROC = fixedDataSizePerHGCROC,
            NData = triggercells
            )
    return producer

