import FWCore.ParameterSet.Config as cms

# helper functions
from HLTrigger.Configuration.common import *


# Enable good edge CPE algorithm in pixel local reconstruction
def customiseFor47966(process):
    for prod in esproducers_by_type(process, 'PixelCPEGenericESProducer', 'PixelCPEFastParamsESProducerAlpakaPhase1@alpaka'):
        if not hasattr(prod, 'GoodEdgeAlgo'):
            prod.GoodEdgeAlgo = cms.bool(True)
        else:
            prod.GoodEdgeAlgo = True

    return process 
    