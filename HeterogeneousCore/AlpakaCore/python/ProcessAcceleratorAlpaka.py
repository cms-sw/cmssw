import FWCore.ParameterSet.Config as cms

import os

from HeterogeneousCore.Common.PlatformStatus import PlatformStatus

class ModuleTypeResolverAlpaka:
    def __init__(self, accelerators, backend):
        # first element is used as the default if nothing is set
        self._valid_backends = []
        if "gpu-nvidia" in accelerators:
            self._valid_backends.append("cuda_async")
        if "gpu-amd" in accelerators:
            self._valid_backends.append("rocm_async")
        if "cpu" in accelerators:
            self._valid_backends.append("serial_sync")
        if len(self._valid_backends) == 0:
            raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "ModuleTypeResolverAlpaka had no backends available because of the combination of the job configuration and accelerator availability of on the machine. The job sees {} accelerators.".format(", ".join(accelerators)))
        if backend is not None:
            if not backend in self._valid_backends:
                raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "The ProcessAcceleratorAlpaka was configured to use {} backend, but that backend is not available because of the combination of the job configuration and accelerator availability on the machine. The job was configured to use {} accelerators, which translates to {} Alpaka backends.".format(
                    backend, ", ".join(accelerators), ", ".join(self._valid_backends)))
            if backend != self._valid_backends[0]:
                self._valid_backends.remove(backend)
                self._valid_backends.insert(0, backend)

    def plugin(self):
        return "ModuleTypeResolverAlpaka"

    def setModuleVariant(self, module):
        if module.type_().endswith("@alpaka"):
            defaultBackend = self._valid_backends[0]
            if hasattr(module, "alpaka"):
                if hasattr(module.alpaka, "backend"):
                    if module.alpaka.backend == "":
                        module.alpaka.backend = defaultBackend
                    elif module.alpaka.backend.value() not in self._valid_backends:
                        raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "Module {} has the Alpaka backend set explicitly, but its accelerator is not available for the job because of the combination of the job configuration and accelerator availability on the machine. The following Alpaka backends are available for the job {}.".format(module.label_(), ", ".join(self._valid_backends)))
                else:
                    module.alpaka.backend = cms.untracked.string(defaultBackend)
            else:
                module.alpaka = cms.untracked.PSet(
                    backend = cms.untracked.string(defaultBackend)
                )

class ProcessAcceleratorAlpaka(cms.ProcessAccelerator):
    """ProcessAcceleratorAlpaka itself does not define or inspect
    availability of any accelerator devices. It merely sets up
    necessary Alpaka infrastructure based on the availability of
    accelerators that the concrete ProcessAccelerators (like
    ProcessAcceleratorCUDA) define.
    """
    def __init__(self):
        super(ProcessAcceleratorAlpaka, self).__init__()
        self._backend = None

    # User-facing interface
    def setBackend(self, backend):
        self._backend = backend

    # Framework-facing interface
    def moduleTypeResolver(self, accelerators):
        return ModuleTypeResolverAlpaka(accelerators, self._backend)

    def apply(self, process, accelerators):
        # Propagate the AlpakaService messages through the MessageLogger
        if not hasattr(process.MessageLogger, "AlpakaService"):
            process.MessageLogger.AlpakaService = cms.untracked.PSet()

        # Check if the CPU backend is available
        try:
            if not "cpu" in accelerators:
                raise False
            from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
        except:
            # the CPU backend is not available, do not load the AlpakaServiceSerialSync
            if hasattr(process, "AlpakaServiceSerialSync"):
                del process.AlpakaServiceSerialSync
        else:
            # the CPU backend is available, ensure the AlpakaServiceSerialSync is loaded
            if not hasattr(process, "AlpakaServiceSerialSync"):
                process.add_(AlpakaServiceSerialSync)

        # Check if CUDA is available, and if the system has at least one usable NVIDIA GPU
        try:
            if not "gpu-nvidia" in accelerators:
                raise False
            from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
        except:
            # CUDA is not available, do not load the AlpakaServiceCudaAsync
            if hasattr(process, "AlpakaServiceCudaAsync"):
                del process.AlpakaServiceCudaAsync
        else:
            # CUDA is available, ensure the AlpakaServiceCudaAsync is loaded
            if not hasattr(process, "AlpakaServiceCudaAsync"):
                process.add_(AlpakaServiceCudaAsync)

        # Check if ROCm is available, and if the system has at least one usable AMD GPU
        try:
            if not "gpu-amd" in accelerators:
                raise False
            from HeterogeneousCore.AlpakaServices.AlpakaServiceROCmAsync_cfi import AlpakaServiceROCmAsync
        except:
            # ROCm is not available, do not load the AlpakaServiceROCmAsync
            if hasattr(process, "AlpakaServiceROCmAsync"):
                del process.AlpakaServiceROCmAsync
        else:
            # ROCm is available, ensure the AlpakaServiceROCmAsync is loaded
            if not hasattr(process, "AlpakaServiceROCmAsync"):
                process.add_(AlpakaServiceROCmAsync)


# Ensure this module is kept in the configuration when dumping it
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorAlpaka, "from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka import ProcessAcceleratorAlpaka")
