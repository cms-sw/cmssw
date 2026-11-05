#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"
#include "PhysicsTools/ONNXRuntime/interface/Backend.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"

namespace cms::Ort {

  namespace {

    constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
    constexpr const char* kMIGraphXExecutionProvider = "MIGraphXExecutionProvider";

    // Register an execution provider library with the ONNX Runtime environment, and return the devices it exposes;
    // the list is empty if there are no suitable devices. A relative library name is looked up in the same directory
    // as libonnxruntime.so .
    std::vector<::Ort::ConstEpDevice> registerExecutionProvider(::Ort::Env& env,
                                                                const char* name,
                                                                const ORTCHAR_T* library) {
      env.RegisterExecutionProviderLibrary(name, library);
      std::vector<::Ort::ConstEpDevice> devices;
      for (const auto& device : env.GetEpDevices()) {
        if (std::string_view(device.EpName()) == name) {
          devices.push_back(device);
        }
      }
      return devices;
    }

    [[noreturn]] void throwUnavailableAccelerator(Backend backend, const char* reason) {
      edm::Exception ex(edm::errors::UnavailableAccelerator);
      ex << backendName(backend) << " backend requested, but " << reason;
      ex.addContext("Calling cms::Ort::ONNXService::configure()");
      throw ex;
    }

  }  // namespace

  // The ONNX Runtime service.
  //
  // It owns the ONNX Runtime environment and registers the execution provider libraries, once per job. Registering
  // them from more than one place is an error: the ONNX Runtime environment is a process-wide singleton, and the
  // second registration under the same name fails with "library is already registered under ...".
  //
  // The service uses the ResourceInformation service to find which backends are available, and the CUDAService and
  // ROCmService - if they are configured - to count the devices. Getting those services in the constructor also
  // orders their lifetimes: they are constructed before this service, and destroyed after it, so the CUDA and HIP
  // runtimes are loaded before the ONNX Runtime uses them, and are reset and unloaded only once it has released
  // them.
  class ONNXService : public ONNXInterface {
  public:
    ONNXService(edm::ParameterSet const& config);
    ~ONNXService() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    bool isAvailable(Backend backend) const final;
    int numberOfDevices(Backend backend) const final;
    int chooseDevice(Backend backend, edm::StreamID id) const final;

    ::Ort::Env& environment() final { return env_; }

    void configure(::Ort::SessionOptions& options, Backend backend, edm::StreamID id) const final;

    ::Ort::SessionOptions sessionOptions(Backend backend, edm::StreamID id) const final;
    ::Ort::SessionOptions sessionOptions(Backend backend, int device = 0, void* compute_stream = nullptr) const final;

  private:
    // Apply the framework settings to the given session options and, for a GPU backend, append the execution provider
    // for the given device, running in the given compute stream, or in its own if the stream is null.
    void configure(::Ort::SessionOptions& options, Backend backend, int device, void* compute_stream) const;

    // The devices exposed by an execution provider, and the number of devices usable by the job.
    struct Provider {
      std::vector<::Ort::ConstEpDevice> devices;
      int number_of_devices = 0;
      bool available() const { return number_of_devices > 0 and not devices.empty(); }
    };

    // The execution provider device matching the given runtime device index, for the CUDA execution provider, which
    // runs on the device added to the session options and stores its index in the "cuda_device_id" metadata.
    ::Ort::ConstEpDevice selectCudaDevice(int device) const;

    // The ONNX Runtime environment.
    // It is a function-local static rather than a data member, so that it is destroyed when the process ends and not
    // when this service is: destroying it unregisters the execution provider libraries and unloads them, and the
    // MIGraphX execution provider and the ROCm runtime crash if that happens while their own static objects are
    // still alive. Note that the logging level is set by the first job that uses it.
    static ::Ort::Env& environmentInstance(bool verbose) {
      static ::Ort::Env env(verbose ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_ERROR, "ONNXService");
      return env;
    }

    // non-const because the execution provider libraries are registered with it; the Ort::Env methods are thread safe
    ::Ort::Env& env_;
    Provider cuda_;
    Provider rocm_;
  };

  ONNXService::ONNXService(edm::ParameterSet const& config)
      : env_(environmentInstance(config.getUntrackedParameter<bool>("verbose"))) {
    // Get the CUDAService and the ROCmService, if they are configured: this makes them outlive this service, so the
    // CUDA and HIP runtimes are unloaded and the devices are reset only after the ONNX Runtime has been torn down.
    // This also constructs them, if they have not been constructed yet: they must be accessed before the
    // ResourceInformation service, because they are the ones that record the available GPUs in it.
    edm::Service<CUDAInterface> cuda;
    edm::Service<ROCmInterface> rocm;
    const bool cuda_enabled = cuda.isAvailable() and cuda->enabled();
    const bool rocm_enabled = rocm.isAvailable() and rocm->enabled();

    // The backends available in the job: the ResourceInformation service knows about the GPUs that are visible to
    // the job and whose use is enabled in the configuration.
    edm::Service<edm::ResourceInformation> resources;
    const bool nvidia = resources.isAvailable() and resources->hasGpuNvidia();
    const bool amd = resources.isAvailable() and resources->hasGpuAMD();

    if (nvidia and cuda_enabled) {
      cuda_.number_of_devices = cuda->numberOfDevices();
      cuda_.devices =
          registerExecutionProvider(env_, kCudaExecutionProvider, ORT_TSTR("libonnxruntime_providers_cuda.so"));
    }

    if (amd and rocm_enabled) {
      rocm_.number_of_devices = rocm->numberOfDevices();
      rocm_.devices =
          registerExecutionProvider(env_, kMIGraphXExecutionProvider, ORT_TSTR("libonnxruntime_providers_migraphx.so"));
    }

    edm::LogInfo message("ONNXService");
    message << "ONNX Runtime " << ::Ort::GetVersionString() << ", CPU backend available";
    if (cuda_.available()) {
      message << ", CUDA backend available on " << cuda_.number_of_devices << " device(s)";
    }
    if (rocm_.available()) {
      message << ", ROCm backend available on " << rocm_.number_of_devices << " device(s)";
    }
  }

  void ONNXService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)->setComment("Log the ONNX Runtime messages at the verbose level.");
    descriptions.add("ONNXService", desc);
  }

  bool ONNXService::isAvailable(Backend backend) const {
    switch (backend) {
      case Backend::cpu:
        return true;
      case Backend::cuda:
        return cuda_.available();
      case Backend::rocm:
        return rocm_.available();
    }
    return false;
  }

  int ONNXService::numberOfDevices(Backend backend) const {
    switch (backend) {
      case Backend::cpu:
        return 1;
      case Backend::cuda:
        return cuda_.available() ? cuda_.number_of_devices : 0;
      case Backend::rocm:
        return rocm_.available() ? rocm_.number_of_devices : 0;
    }
    return 0;
  }

  int ONNXService::chooseDevice(Backend backend, edm::StreamID id) const {
    const int devices = numberOfDevices(backend);
    if (devices < 1) {
      throwUnavailableAccelerator(backend, "it is not available in this job");
    }
    // Distribute the framework streams over the devices, like the alpaka modules do. This is suboptimal if the
    // number of streams is not a multiple of the number of devices, and does no load balancing.
    return id % devices;
  }

  ::Ort::ConstEpDevice ONNXService::selectCudaDevice(int device) const {
    const std::string id = std::to_string(device);
    for (const auto& ep_device : cuda_.devices) {
      const char* value = ep_device.EpMetadata().GetValue("cuda_device_id");
      if (value and id == value) {
        return ep_device;
      }
    }

    edm::Exception ex(edm::errors::UnavailableAccelerator);
    ex << "CUDA backend requested for device " << device << ", but the ONNX Runtime " << kCudaExecutionProvider
       << " provides only the devices";
    for (const auto& ep_device : cuda_.devices) {
      const char* value = ep_device.EpMetadata().GetValue("cuda_device_id");
      ex << ' ' << (value ? value : "(unknown)");
    }
    ex.addContext("Calling cms::Ort::ONNXService::configure()");
    throw ex;
  }

  void ONNXService::configure(::Ort::SessionOptions& options, Backend backend, edm::StreamID id) const {
    configure(options, backend, chooseDevice(backend, id), nullptr);
  }

  ::Ort::SessionOptions ONNXService::sessionOptions(Backend backend, edm::StreamID id) const {
    ::Ort::SessionOptions options;
    configure(options, backend, id);
    return options;
  }

  ::Ort::SessionOptions ONNXService::sessionOptions(Backend backend, int device, void* compute_stream) const {
    ::Ort::SessionOptions options;
    configure(options, backend, device, compute_stream);
    return options;
  }

  void ONNXService::configure(::Ort::SessionOptions& options, Backend backend, int device, void* compute_stream) const {
    // Disable the ONNX Runtime internal threading model: all the CPU based operations run single-threaded, in the
    // calling thread, so that the framework keeps control of the number of threads used by the job.
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (backend == Backend::cpu) {
      return;
    }
    if (not isAvailable(backend)) {
      throwUnavailableAccelerator(backend, "it is not available in this job");
    }

    // The GPU execution providers are built as plugin libraries, see
    // https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/ : they are registered with the ONNX
    // Runtime environment by the constructor of this service, and the device used by the session is chosen among the
    // devices they expose.
    // Both execution providers support running in a stream provided by the user, with the same options. Note that
    // the options are validated by the execution providers, and the MIGraphX one terminates the process if it finds
    // an option it does not know, so the other options are set separately for each backend.
    std::unordered_map<std::string, std::string> provider_options;
    if (compute_stream != nullptr) {
      provider_options["has_user_compute_stream"] = "1";
      provider_options["user_compute_stream"] = std::to_string(reinterpret_cast<std::uintptr_t>(compute_stream));
    }

    if (backend == Backend::cuda) {
      // The CUDA execution provider runs on the device added to the session options, and ignores the "device_id"
      // option; the device is identified by the "cuda_device_id" entry of its metadata.
      provider_options["device_id"] = std::to_string(device);
      if (compute_stream != nullptr) {
        // Run the memory copies in the compute stream: the queues used by the alpaka modules are non-blocking, so
        // they would not be synchronised with the legacy default stream.
        provider_options["do_copy_in_default_stream"] = "0";
        // Grow the memory arena only by the amount requested ("1" is kSameAsRequested). The "arena." options are
        // forwarded by the plugin execution provider to the allocator it creates.
        provider_options["arena.extend_strategy"] = "1";
      }
      options.AppendExecutionProvider_V2(env_, {selectCudaDevice(device)}, provider_options);
    } else if (backend == Backend::rocm) {
      // The MIGraphX execution provider ignores the device added to the session options, and calls hipSetDevice()
      // with the device given by the "device_id" option, which can be any device of the HIP runtime. It validates
      // the device, and checks that the user compute stream belongs to it, while creating the session, inside a
      // non-throwing function, where an error terminates the process.
      if (device < 0 or device >= rocm_.number_of_devices) {
        edm::Exception ex(edm::errors::UnavailableAccelerator);
        ex << "ROCm backend requested for device " << device << ", but the job can use only " << rocm_.number_of_devices
           << " device(s)";
        ex.addContext("Calling cms::Ort::ONNXService::configure()");
        throw ex;
      }
      // The MIGraphX execution provider always runs the memory copies in the compute stream, and does not support the
      // arena options.
      provider_options["device_id"] = std::to_string(device);
      options.AppendExecutionProvider_V2(env_, {rocm_.devices.front()}, provider_options);
    }
  }

}  // namespace cms::Ort

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using ONNXServiceMaker = edm::serviceregistry::ParameterSetMaker<cms::Ort::ONNXInterface, cms::Ort::ONNXService>;
DEFINE_FWK_SERVICE_MAKER(ONNXService, ONNXServiceMaker);
