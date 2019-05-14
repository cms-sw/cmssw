import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc


def custom_triggercellselect_supertriggercell(process,
                                              stcSize=supertc_conc_proc.stcSize
                                              ):
    parameters = supertc_conc_proc.clone(stcSize = stcSize)
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_threshold(process,
                                       threshold_silicon=threshold_conc_proc.triggercell_threshold_silicon,  # in mipT
                                       threshold_scintillator=threshold_conc_proc.triggercell_threshold_scintillator  # in mipT
                                       ):
    parameters = threshold_conc_proc.clone(
            triggercell_threshold_silicon = threshold_silicon,
            triggercell_threshold_scintillator = threshold_scintillator
            )
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_bestchoice(process,
                                        triggercells=best_conc_proc.NData
                                        ):
    parameters = best_conc_proc.clone(NData = triggercells)
    process.hgcalConcentratorProducer.ProcessorParameters = parameters
    return process
