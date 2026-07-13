// CUPTI kernel-launch logger service.
//
// An EDM service that records which CUDA kernels a cmsRun job launches and which module
// launched each of them. cudaKernelStackBudget --launched adds it to the configuration and
// reads its output; it can also be added to any configuration directly:
//
//   process.CuptiKernelLoggerService = cms.Service("CuptiKernelLoggerService",
//       kernelLog  = cms.untracked.string("kernels.txt"),    # "<mangled kernel>\t<module>\t<dyn shared>" lines
//       libraryLog = cms.untracked.string("libraries.txt"))  # one loaded library path per line
//
// It subscribes to the CUPTI launch callbacks (runtime and driver API) and, for each launch,
// records the kernel together with the module running on the calling thread and the largest amount
// of dynamic shared memory the launch requested. CUPTI allows a single subscriber per process, so
// if another CUPTI client (NVProfilerService, nsys, ncu) is already active the subscription fails
// and the service logs a warning explaining why its logs will be empty.

#include <cstddef>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <link.h>  // dl_iterate_phdr

#include <cuda_runtime_api.h>  // cudaLaunchConfig_t (dynamicSmemBytes)
#include <cupti.h>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

namespace {
  std::mutex g_mutex;
  // (mangled kernel name, module label) -> largest dynamic shared memory (bytes) requested by any
  // launch of that kernel from that module. Static shared memory is a compile-time property read
  // separately with cuobjdump; the dynamic amount is only known here, at launch time.
  std::map<std::pair<std::string, std::string_view>, std::size_t> g_records;
  // modules currently executing acquire()/produce() on this host thread; a kernel launch fires
  // its callback synchronously on the same thread, so the top of the stack is the module that
  // issued the launch (empty when a kernel is launched outside any module)
  thread_local std::vector<std::string_view> t_modules;

  // dynamic shared memory (bytes) requested by a launch, read from the callback's function
  // arguments. The field lives at a different place in each entry point's parameter struct (and
  // __cudaLaunchKernel exposes no typed struct at all, so it reports 0); this is harmless because a
  // single launch also surfaces through cudaLaunchKernel/cuLaunchKernel, whose structs do carry it,
  // and g_records keeps the maximum across all of a kernel's launches.
  std::size_t dynamicSharedBytes(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* params) {
    if (params == nullptr) {
      return 0;
    }
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
      switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
          return static_cast<const cudaLaunchKernel_v7000_params*>(params)->sharedMem;
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
          return static_cast<const cudaLaunchCooperativeKernel_v9000_params*>(params)->sharedMem;
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060: {
          auto const* p = static_cast<const cudaLaunchKernelExC_v11060_params*>(params);
          return p->config ? p->config->dynamicSmemBytes : 0;
        }
        default:  // e.g. __cudaLaunchKernel: no parameter struct to read
          return 0;
      }
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
      switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
          return static_cast<const cuLaunchKernel_params*>(params)->sharedMemBytes;
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
          return static_cast<const cuLaunchCooperativeKernel_params*>(params)->sharedMemBytes;
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
          auto const* p = static_cast<const cuLaunchKernelEx_params*>(params);
          return p->config ? p->config->sharedMemBytes : 0;
        }
        default:
          return 0;
      }
    }
    return 0;
  }

  void CUPTIAPI launchCallback(void* /* userdata */,
                               CUpti_CallbackDomain domain,
                               CUpti_CallbackId cbid,
                               const void* callbackData) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API) {
      return;
    }
    auto const* data = static_cast<const CUpti_CallbackData*>(callbackData);
    if (data->callbackSite != CUPTI_API_ENTER || data->symbolName == nullptr) {
      return;
    }
    std::size_t dynamicShared = dynamicSharedBytes(domain, cbid, data->functionParams);
    std::string_view module = t_modules.empty() ? std::string_view() : t_modules.back();
    std::lock_guard<std::mutex> guard(g_mutex);
    // Each launch can surface through several of the enabled entry points (a runtime cudaLaunchKernel
    // dispatches to the driver cuLaunchKernel, both firing here), but keying the map on
    // (kernel, module) and keeping the maximum dynamic shared memory collapses those into one entry.
    std::size_t& maxDynamicShared = g_records[std::pair<std::string, std::string_view>(data->symbolName, module)];
    if (dynamicShared > maxDynamicShared) {
      maxDynamicShared = dynamicShared;
    }
  }

  int collectLoadedObject(struct dl_phdr_info* info, size_t /* size */, void* data) {
    if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') {
      static_cast<std::set<std::string>*>(data)->emplace(info->dlpi_name);
    }
    return 0;
  }
}  // namespace

class CuptiKernelLoggerService {
public:
  CuptiKernelLoggerService(edm::ParameterSet const&, edm::ActivityRegistry&);
  ~CuptiKernelLoggerService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void pushModule(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void popModule(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void writeLogs();

  const std::string kernelLog_;
  const std::string libraryLog_;

  CUpti_SubscriberHandle subscriber_ = nullptr;
};

CuptiKernelLoggerService::CuptiKernelLoggerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : kernelLog_(config.getUntrackedParameter<std::string>("kernelLog", "")),
      libraryLog_(config.getUntrackedParameter<std::string>("libraryLog", "")) {
  // writeLogs needs no CUPTI (libraryLog comes from dl_iterate_phdr), so register it
  // unconditionally; the kernel records simply stay empty if the subscription below fails
  registry.watchPostEndJob(this, &CuptiKernelLoggerService::writeLogs);

  CUptiResult result = cuptiSubscribe(&subscriber_, reinterpret_cast<CUpti_CallbackFunc>(launchCallback), nullptr);
  if (result != CUPTI_SUCCESS) {
    subscriber_ = nullptr;  // leave no handle for the destructor to unsubscribe
    if (result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED || result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      edm::LogWarning("CuptiKernelLoggerService")
          << "Another CUPTI client is already active in this process (for example NVProfilerService, "
             "nsys or ncu). CUPTI allows only one subscriber per process, so CuptiKernelLoggerService "
             "could not attach and its logs will be empty. Remove the other CUPTI client and rerun.";
    } else {
      const char* message = nullptr;
      cuptiGetResultString(result, &message);
      edm::LogWarning("CuptiKernelLoggerService") << "cuptiSubscribe failed (" << (message ? message : "unknown error")
                                                  << "); CuptiKernelLoggerService logs will be empty.";
    }
    return;
  }

  // enable every kernel-launch entry point; the launch path used depends on the Alpaka/CUDA code.
  // each runtime/driver entry point exposes a distinct callback id for the default-stream "legacy"
  // flavour and for the "_ptsz" per-thread-default-stream flavour selected by nvcc
  // --default-stream per-thread, so enable both to cover either compilation mode. Overlapping
  // callbacks do not cause duplicate records: g_records is a map keyed on (kernel, module).
  // a failed enable would silently drop that path, so warn rather than report an empty log later
  auto enable = [&](CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    CUptiResult status = cuptiEnableCallback(1, subscriber_, domain, cbid);
    if (status != CUPTI_SUCCESS) {
      const char* message = nullptr;
      cuptiGetResultString(status, &message);
      edm::LogWarning("CuptiKernelLoggerService")
          << "cuptiEnableCallback failed for callback id " << cbid << " (" << (message ? message : "unknown error")
          << "); launches through that entry point will not be recorded.";
    }
  };
  // runtime API (cudaLaunch*); __cudaLaunchKernel is the internal entry point CUDA 13 dispatches to
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000);
  // driver API (cuLaunch*)
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice);

  // bracket the host phases where kernels are launched, so the callback can attribute each
  // launch to the running module: acquire() for ExternalWork modules, produce() for the rest
  registry.watchPreModuleEventAcquire(this, &CuptiKernelLoggerService::pushModule);
  registry.watchPostModuleEventAcquire(this, &CuptiKernelLoggerService::popModule);
  registry.watchPreModuleEvent(this, &CuptiKernelLoggerService::pushModule);
  registry.watchPostModuleEvent(this, &CuptiKernelLoggerService::popModule);

  edm::LogInfo("CuptiKernelLoggerService") << "CUPTI kernel logger service successfully initialised." << '\n'
                                           << "Launched kernels will be written to '" << kernelLog_ << "'\n"
                                           << "Loaded libraries will be written to '" << libraryLog_ << "'.";
}

CuptiKernelLoggerService::~CuptiKernelLoggerService() {
  if (subscriber_ != nullptr) {
    cuptiUnsubscribe(subscriber_);
  }
}

void CuptiKernelLoggerService::pushModule(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  t_modules.push_back(std::string_view(mcc.moduleDescription()->moduleLabel()));
}

void CuptiKernelLoggerService::popModule(edm::StreamContext const&, edm::ModuleCallingContext const&) {
  if (not t_modules.empty()) {
    t_modules.pop_back();
  }
}

void CuptiKernelLoggerService::writeLogs() {
  std::lock_guard<std::mutex> guard(g_mutex);
  if (not kernelLog_.empty()) {
    std::ofstream out(kernelLog_);
    for (auto const& [key, dynamicShared] : g_records) {
      auto const& [kernel, module] = key;
      out << kernel << '\t' << module << '\t' << dynamicShared << '\n';
    }
  }
  if (not libraryLog_.empty()) {
    std::set<std::string> libraries;
    // loop over the loaded shared objects and calls their callback
    dl_iterate_phdr(collectLoadedObject, &libraries);
    std::ofstream out(libraryLog_);
    for (auto const& library : libraries) {
      out << library << '\n';
    }
  }
}

void CuptiKernelLoggerService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("CUPTI kernel-launch logger service");
  desc.addUntracked<std::string>("kernelLog", "kernels.txt")
      ->setComment(
          "Path to a text file where the service will write "
          "<mangled kernel>\\t<module>\\t<max dynamic shared bytes> lines");
  desc.addUntracked<std::string>("libraryLog", "libraries.txt")
      ->setComment("Path to a text file where the service will write one loaded library path per line");
  descriptions.add("CuptiKernelLoggerService", desc);
}

DEFINE_FWK_SERVICE(CuptiKernelLoggerService);
