import FWCore.ParameterSet.Config as cms

import os

from HeterogeneousCore.Common.PlatformStatus import PlatformStatus

class ProcessAcceleratorROCm(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorROCm, self).__init__()
        self._label = "gpu-amd"

    def labels(self):
        return [ self._label ]

    def enabledLabels(self):
        # Check if ROCm is available, and if the system has at least one usable device.
        # These should be checked on each worker node, because it depends both
        # on the architecture and on the actual hardware present in the machine.
        status = PlatformStatus(os.waitstatus_to_exitcode(os.system("rocmIsEnabled")))
        return self.labels() if status == PlatformStatus.Success else []

    def apply(self, process, accelerators):

        if self._label in accelerators:
            # Ensure that the ROCmService is loaded
            if not hasattr(process, "ROCmService"):
                from HeterogeneousCore.ROCmServices.ROCmService_cfi import ROCmService
                process.add_(ROCmService)

            # Propagate the ROCmService messages through the MessageLogger
            if not hasattr(process.MessageLogger, "ROCmService"):
                process.MessageLogger.ROCmService = cms.untracked.PSet()

        else:
            # Make sure the ROCmService is not loaded
            if hasattr(process, "ROCmService"):
                del process.ROCmService

            # Drop the ROCmService messages from the MessageLogger
            if hasattr(process.MessageLogger, "ROCmService"):
                del process.MessageLogger.ROCmService


# Ensure this module is kept in the configuration when dumping it
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorROCm, "from HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm import ProcessAcceleratorROCm")
