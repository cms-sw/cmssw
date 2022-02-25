import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.gpu_cff import *
from Configuration.ProcessModifiers.gpuValidationEcal_cff import *

# This modifier chain is for turning on DQM modules used for gpu validation

gpuValidation =  cms.ModifierChain(
    gpu,
    gpuValidationEcal
)
