import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# random numbers initialization service
from IOMC.RandomEngine.IOMC_cff import *

# DQM store service
from DQMServices.Core.DQMStore_cfi import *

# load CUDA services when the "gpu" or "pixelNtupletFit" modifiers are enabled
def _addCUDAServices(process):
     process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
modifyConfigurationStandardSequencesServicesAddCUDAServices_ = (gpu | pixelNtupletFit).makeProcessModifier(_addCUDAServices)

# load TritonService when SONIC workflow is enabled
def _addTritonService(process):
	process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
modifyConfigurationStandardSequencesServicesAddTritonService_ = enableSonicTriton.makeProcessModifier(_addTritonService)
