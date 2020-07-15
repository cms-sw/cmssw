import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import vfe_proc

def create_compression(process,
       exponent=vfe_proc.exponentBits,
       mantissa=vfe_proc.mantissaBits,
       rounding=vfe_proc.rounding,
       oot_coefficients=vfe_proc.oot_coefficients
        ):
    producer = process.hgcalVFEProducer.clone(
        ProcessorParameters = vfe_proc.clone(
            exponentBits = exponent,
            mantissaBits = mantissa,
            rounding = rounding,
            oot_coefficients = oot_coefficients
        )
    )
    return producer
