import FWCore.ParameterSet.Config as cms


def create_compression(process,
       exponent=4,
       mantissa=4,
       rounding=True
        ):
    producer = process.hgcalVFEProducer.clone() 
    producer.ProcessorParameters.exponentBits = cms.uint32(exponent)
    producer.ProcessorParameters.mantissaBits = cms.uint32(mantissa)
    producer.ProcessorParameters.rounding = cms.bool(rounding)
    return producer
