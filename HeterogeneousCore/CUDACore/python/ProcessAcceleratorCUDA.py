import FWCore.ParameterSet.Config as cms

import os
import sys

from HeterogeneousCore.Common.PlatformStatus import PlatformStatus

class ProcessAcceleratorCUDA(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorCUDA, self).__init__()
        self._label = "gpu-nvidia"

    def labels(self):
        return [ self._label ]

    def enabledLabels(self):
        # Check if CUDA is available, and if the system has at least one usable device.
        # These should be checked on each worker node, because it depends both
        # on the architecture and on the actual hardware present in the machine.
        status = PlatformStatus(os.waitstatus_to_exitcode(os.system("cudaIsEnabled")))
        return self.labels() if status == PlatformStatus.Success else []

    def apply(self, process, accelerators):

        if self._label in accelerators:
            # Ensure that the CUDAService is loaded
            if not hasattr(process, "CUDAService"):
                from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
                process.add_(CUDAService)

            # Propagate the CUDAService messages through the MessageLogger
            if not hasattr(process.MessageLogger, "CUDAService"):
                process.MessageLogger.CUDAService = cms.untracked.PSet()

            # Configure Open MPI 5 to use the CUDA accelerator framework
            if os.environ['OMPI_MCA_accelerator'] == 'null':
                os.environ['OMPI_MCA_accelerator'] = 'cuda'

        else:
            # Make sure the CUDAService is not loaded
            if hasattr(process, "CUDAService"):
                del process.CUDAService

            # Drop the CUDAService messages from the MessageLogger
            if hasattr(process.MessageLogger, "CUDAService"):
                del process.MessageLogger.CUDAService

            # Configure Open MPI 5 to not use the CUDA accelerator framework
            if 'cuda' in os.environ['OMPI_MCA_accelerator']:
                mpiacc = os.environ['OMPI_MCA_accelerator'].split(',')
                mpiacc = [ acc for acc in mpiacc if acc != 'cuda' ] or [ 'null' ]
                os.environ['OMPI_MCA_accelerator'] = ','.join(mpiacc)
                print(f"The 'cuda' accelerator is disabled, 'OMPI_MCA_accelerator' has been set to '{','.join(mpiacc)}'", file=sys.stderr)

# Ensure this module is kept in the configuration when dumping it
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorCUDA, "from HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA import ProcessAcceleratorCUDA")
