import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import vfe_proc

def custom_hgcroc_oot(process,
                      oot_coefficients=vfe_proc.oot_coefficients
                      ):
    parameters = vfe_proc.clone(oot_coefficients = oot_coefficients)
    process.hgcalVFEProducer.ProcessorParameters = parameters
    return process
