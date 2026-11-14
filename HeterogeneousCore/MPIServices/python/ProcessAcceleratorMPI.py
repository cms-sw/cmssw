import FWCore.ParameterSet.Config as cms

import os

class ProcessAcceleratorMPI(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorMPI, self).__init__()

    def apply(self, process, accelerators):
        # Open MPI supports a single accelerator backend, either CUDA or ROCm.
        # What accelerator to use can be autodetected, or selected with the OMPI_MCA_accelerator
        # environment variable.
        # By default, the CMSSW environment disables accelerator support setting it to 'null'.

        # Update the environment only if the MPIService is part of the configuration.
        if not hasattr(process, "MPIService"):
            return

        # Select the Open MPI accelerator backend based on the CMSSW accelerator.
        if "gpu-nvidia" in accelerators:
            os.environ['OMPI_MCA_accelerator'] = 'cuda'
        elif "gpu-amd" in accelerators:
            os.environ['OMPI_MCA_accelerator'] = 'rocm'
        else:
            os.environ['OMPI_MCA_accelerator'] = 'null'


# Ensure this module is kept in the configuration when dumping it
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorMPI, "from HeterogeneousCore.MPIServices.ProcessAcceleratorMPI import ProcessAcceleratorMPI")
