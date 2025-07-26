import FWCore.ParameterSet.Config as cms

# helper functions
from HLTrigger.Configuration.common import *


# Disable the IrradiationBiasCorrection in the Pixel CPE generic reconstruction
def customiseFor48541(process):
    for prod in esproducers_by_type(process, 'PixelCPEGenericESProducer'):
        prod.IrradiationBiasCorrection = False

    return process
