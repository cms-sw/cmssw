import FWCore.ParameterSet.Config as cms

import os

class ProcessAcceleratorROCm(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorROCm,self).__init__()
        self._label = "gpu-amd"
    def labels(self):
        return [self._label]
    def enabledLabels(self):
        enabled = (os.system("rocmIsEnabled") == 0)
        if enabled:
            return self.labels()
        else:
            return []
    def apply(self, process, accelerators):
        if not hasattr(process, "ROCmService"):
            from HeterogeneousCore.ROCmServices.ROCmService_cfi import ROCmService
            process.add_(ROCmService)

        if not hasattr(process.MessageLogger, "ROCmService"):
            process.MessageLogger.ROCmService = cms.untracked.PSet()

        if self._label in accelerators:
            process.ROCmService.enabled = True
        else:
            process.ROCmService.enabled = False
            
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorROCm, "from HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm import ProcessAcceleratorROCm")
