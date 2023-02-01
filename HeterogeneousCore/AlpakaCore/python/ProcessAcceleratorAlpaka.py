import FWCore.ParameterSet.Config as cms

class ModuleTypeResolverAlpaka:
    def __init__(self, accelerators, backend):
        # first element is used as the default is nothing is set
        self._valid_backends = []
        if "gpu-nvidia" in accelerators:
            self._valid_backends.append("cuda_async")
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
        super(ProcessAcceleratorAlpaka,self).__init__()
        self._backend = None
    # User-facing interface
    def setBackend(self, backend):
        self._backend = backend
    # Framework-facing interface
    def moduleTypeResolver(self, accelerators):
        return ModuleTypeResolverAlpaka(accelerators, self._backend)
    def apply(self, process, accelerators):
        if not hasattr(process, "AlpakaServiceSerialSync"):
            from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
            process.add_(AlpakaServiceSerialSync)
        if not hasattr(process, "AlpakaServiceCudaAsync"):
            from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
            process.add_(AlpakaServiceCudaAsync)

        if not hasattr(process.MessageLogger, "AlpakaService"):
            process.MessageLogger.AlpakaService = cms.untracked.PSet()

        process.AlpakaServiceSerialSync.enabled = "cpu" in accelerators
        process.AlpakaServiceCudaAsync.enabled = "gpu-nvidia" in accelerators

cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorAlpaka, "from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka import ProcessAcceleratorAlpaka")
