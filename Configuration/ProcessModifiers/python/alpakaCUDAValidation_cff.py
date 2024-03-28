import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.alpaka_cff import *
from Configuration.ProcessModifiers.alpakaValidationPixel_cff import *
from Configuration.ProcessModifiers.alpakaCUDAValidationPixel_cff import *
from Configuration.ProcessModifiers.gpu_cff import *

# This modifier chain is for turning on DQM modules used for alpaka device/host validation

alpakaCUDAValidation =  cms.ModifierChain(
    alpaka,
    alpakaValidationPixel,
    alpakaCUDAValidationPixel
)

