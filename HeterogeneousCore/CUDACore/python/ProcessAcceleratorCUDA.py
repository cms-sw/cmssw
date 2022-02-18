import FWCore.ParameterSet.Config as cms

import os

class ProcessAcceleratorCUDA(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorCUDA,self).__init__()
        self._label = "gpu-nvidia"
    def labels(self):
        return [self._label]
    def enabledLabels(self):
        enabled = (os.system("cudaIsEnabled") == 0)
        if enabled:
            return self.labels()
        else:
            return []
    def apply(self, process, accelerators):
        if not hasattr(process, "CUDAService"):
            from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
            process.add_(CUDAService)

        if self._label in accelerators:
            process.CUDAService.enabled = True
            process.MessageLogger.CUDAService = cms.untracked.PSet()
        else:
            process.CUDAService.enabled = False
            
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorCUDA, "from HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA import ProcessAcceleratorCUDA")
