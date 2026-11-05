#ifndef PhysicsTools_ONNXRuntime_interface_ONNXInterface_h
#define PhysicsTools_ONNXRuntime_interface_ONNXInterface_h

#include <concepts>
#include <initializer_list>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "FWCore/Utilities/interface/StreamID.h"
#include "PhysicsTools/ONNXRuntime/interface/Backend.h"

namespace cms::Ort {

  // Interface to the ONNXService, which owns the ONNX Runtime environment, registers the execution provider
  // libraries, and knows which backends and devices can be used in the job.
  //
  // The service is required by every module that uses the ONNX Runtime, on any backend, and is accessed with
  // edm::Service<cms::Ort::ONNXInterface>.
  class ONNXInterface {
  public:
    ONNXInterface() = default;
    virtual ~ONNXInterface() = default;

    // True if the given backend can be used in this job: the hardware is present, its use is enabled in the
    // configuration, and the ONNX Runtime execution provider for it exposes at least one device.
    virtual bool isAvailable(Backend backend) const = 0;

    // The number of devices that can be used with the given backend, 1 for Backend::cpu and 0 if the backend is not
    // available.
    virtual int numberOfDevices(Backend backend) const = 0;

    // The first of the given backends that is available in this job, in the order given, e.g.
    // chooseBackend(Backend::cuda, Backend::cpu). Without arguments, the first available of Backend::cuda,
    // Backend::rocm and Backend::cpu. An edm::Exception with category UnavailableAccelerator is thrown if none of the
    // given backends is available.
    template <typename... Backends>
      requires(sizeof...(Backends) > 0 and (std::same_as<Backends, Backend> and ...))
    Backend chooseBackend(Backends... backends) const {
      for (Backend backend : {backends...}) {
        if (isAvailable(backend)) {
          return backend;
        }
      }
      throwUnavailableBackends({backends...});
    }

    Backend chooseBackend() const { return chooseBackend(Backend::cuda, Backend::rocm, Backend::cpu); }

    // The device used by the given framework stream. The streams are distributed over the devices round-robin, the
    // same way the alpaka modules do it in HeterogeneousCore/AlpakaCore/src/alpaka/chooseDevice.cc .
    virtual int chooseDevice(Backend backend, edm::StreamID id) const = 0;

    // The ONNX Runtime environment, shared by all the sessions in the job.
    virtual ::Ort::Env& environment() = 0;

    // Configure the given session options to run the given backend on the device chosen for the given framework
    // stream: apply the framework settings, which disable the ONNX Runtime internal threading, and for a GPU backend
    // append the execution provider for that device.
    // The options can already contain the settings specific to a model or to a module (e.g. the graph optimisation
    // level), but not an execution provider. The framework settings take precedence over the module ones.
    virtual void configure(::Ort::SessionOptions& options, Backend backend, edm::StreamID id) const = 0;

    // The session options to run the given backend on the device chosen for the given framework stream.
    // The session creates and owns its own compute stream, so two sessions built with these options run in two
    // different streams: build one session per module and per framework stream, and keep it for the lifetime of the
    // job, like the alpaka modules do with their queue. SessionCache does this on a GPU, and shares a single session
    // between all the framework streams on the CPU.
    virtual ::Ort::SessionOptions sessionOptions(Backend backend, edm::StreamID id) const = 0;

    // The session options to run the given backend on the given device, in a compute stream owned by the caller
    // (e.g. the stream underlying an alpaka queue) rather than by the session. A null stream lets the session create
    // and own its own.
    // Note that the device is the runtime index of the GPU as seen by the job, i.e. after applying
    // CUDA_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES; it is ignored by Backend::cpu, which can omit it.
    virtual ::Ort::SessionOptions sessionOptions(Backend backend,
                                                 int device = 0,
                                                 void* compute_stream = nullptr) const = 0;

  private:
    // Throw an edm::Exception with category UnavailableAccelerator, listing the given backends.
    [[noreturn]] static void throwUnavailableBackends(std::initializer_list<Backend> backends);
  };

}  // namespace cms::Ort

#endif  // PhysicsTools_ONNXRuntime_interface_ONNXInterface_h
