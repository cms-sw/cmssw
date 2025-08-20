import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.alpaka_cff import alpaka

#
# Load all Pixel Cluster Position Estimator ESProducers
#
# 1. Template algorithm
#
from RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi import *
#
# 2. Pixel Generic CPE
#
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
#
# 3. ESProducer for the Magnetic-field dependent template records
#
from CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi import *
from CalibTracker.SiPixelESProducers.SiPixel2DTemplateDBObjectESProducer_cfi import *

def _addProcessCPEsAlpaka(process):
    process.load("RecoLocalTracker.SiPixelRecHits.pixelCPEFastParamsESProducerAlpakaPhase1_cfi")
    process.load("RecoLocalTracker.SiPixelRecHits.pixelCPEFastParamsESProducerAlpakaPhase2_cfi")
    process.load("RecoLocalTracker.SiPixelRecHits.pixelCPEFastParamsESProducerAlpakaHIonPhase1_cfi")

modifyConfigurationForAlpakaCPEs_ = alpaka.makeProcessModifier(_addProcessCPEsAlpaka)

