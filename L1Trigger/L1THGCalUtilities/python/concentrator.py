import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc


def create_supertriggercell(process, inputs,
                            stcSize=supertc_conc_proc.stcSize
                            ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = supertc_conc_proc.clone(
            stcSize = stcSize
            )
    return producer


def create_threshold(process, inputs,
                     threshold_silicon=threshold_conc_proc.triggercell_threshold_silicon,  # in mipT
                     threshold_scintillator=threshold_conc_proc.triggercell_threshold_scintillator  # in mipT
                     ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = threshold_conc_proc.clone(
            triggercell_threshold_silicon = threshold_silicon,  # MipT
            triggercell_threshold_scintillator = threshold_scintillator  # MipT
            )
    return producer


def create_bestchoice(process, inputs,
                      triggercells=best_conc_proc.NData
                      ):
    producer = process.hgcalConcentratorProducer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs)),
            InputTriggerSums = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    producer.ProcessorParameters = best_conc_proc.clone(
            NData = triggercells
            )
    return producer
