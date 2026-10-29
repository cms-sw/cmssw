import FWCore.ParameterSet.Config as cms

class ProcessAcceleratorPyTorchAlpaka(cms.ProcessAccelerator):
    """Enable the CMSSW allocator bridge for selected GPU backends."""

    def apply(self, process, accelerators):
        if "gpu-nvidia" in accelerators:
            if not hasattr(process, "PyTorchAlpakaServiceCudaAsync"):
                from PhysicsTools.PyTorchAlpaka.PyTorchAlpakaServiceCudaAsync_cfi import (
                    PyTorchAlpakaServiceCudaAsync,
                )
                process.add_(PyTorchAlpakaServiceCudaAsync)

        elif "gpu-amd" in accelerators:
            if not hasattr(process, "PyTorchAlpakaServiceROCmAsync"):
                from PhysicsTools.PyTorchAlpaka.PyTorchAlpakaServiceROCmAsync_cfi import (
                    PyTorchAlpakaServiceROCmAsync,
                )
                process.add_(PyTorchAlpakaServiceROCmAsync)

        if "gpu-nvidia" in accelerators or "gpu-amd" in accelerators:
            if not hasattr(process.MessageLogger, "PyTorchAlpakaService"):
                process.MessageLogger.PyTorchAlpakaService = cms.untracked.PSet()


cms.specialImportRegistry.registerSpecialImportForType(
    ProcessAcceleratorPyTorchAlpaka,
    "from PhysicsTools.PyTorchAlpaka.ProcessAcceleratorPyTorchAlpaka "
    "import ProcessAcceleratorPyTorchAlpaka",
)
