#ifndef PhysicsTools_ONNXRuntime_interface_Backend_h
#define PhysicsTools_ONNXRuntime_interface_Backend_h

namespace cms::Ort {

  enum class Backend {
    cpu,
    cuda,  // NVIDIA GPUs, using the CUDA execution provider
    rocm,  // AMD GPUs, using the MIGraphX execution provider
           // note: MIGraphX recompiles the model whenever the input shapes change from one call to the next,
           // so it is best suited for models with fixed input shapes (e.g. a fixed or padded batch size)
  };

  // The name of the backend, for use in messages.
  inline const char* backendName(Backend backend) {
    switch (backend) {
      case Backend::cpu:
        return "CPU";
      case Backend::cuda:
        return "CUDA";
      case Backend::rocm:
        return "ROCm";
    }
    return "unknown";
  }

}  // namespace cms::Ort

#endif  // PhysicsTools_ONNXRuntime_interface_Backend_h
