import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# random numbers initialization service
from IOMC.RandomEngine.IOMC_cff import *

# DQM store service
from DQMServices.Core.DQMStore_cfi import *

# load ProcessAccelerators (that set the e.g. the necessary CUDA
# stuff) when the "gpu" or "pixelNtupletFit" modifiers are enabled
def _addProcessAccelerators(process):
    process.load("Configuration.StandardSequences.Accelerators_cff")

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
modifyConfigurationStandardSequencesServicesAddProcessAccelerators_ = (gpu | pixelNtupletFit).makeProcessModifier(_addProcessAccelerators)

# load TritonService when SONIC workflow is enabled
def _addTritonService(process):
	process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
modifyConfigurationStandardSequencesServicesAddTritonService_ = enableSonicTriton.makeProcessModifier(_addTritonService)
