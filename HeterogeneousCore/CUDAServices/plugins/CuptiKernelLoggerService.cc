// CUPTI kernel-launch logger service.
//
// An EDM service that records which CUDA kernels a cmsRun job launches and which module
// launched each of them. cudaKernelStackBudget --launched adds it to the configuration and
// reads its output; it can also be added to any configuration directly:
//
//   process.CuptiKernelLoggerService = cms.Service("CuptiKernelLoggerService",
//       kernelLog  = cms.untracked.string("kernels.txt"),    # "<mangled kernel>\t<module>" lines
//       libraryLog = cms.untracked.string("libraries.txt"))  # one loaded library path per line
//
// It subscribes to the CUPTI launch callbacks (runtime and driver API) and, for each launch,
// records the kernel together with the module running on the calling thread. CUPTI allows a
// single subscriber per process, so if another CUPTI client (NVProfilerService, nsys, ncu)
// is already active the subscription fails and the service logs a warning explaining why its
// logs will be empty.

#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <link.h>  // dl_iterate_phdr

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
  // (mangled kernel name, module label) pairs seen during the job
  std::set<std::pair<std::string, std::string>> g_records;
  // modules currently executing acquire()/produce() on this host thread; a kernel launch fires
  // its callback synchronously on the same thread, so the top of the stack is the module that
  // issued the launch (empty when a kernel is launched outside any module)
  thread_local std::vector<std::string> t_modules;

  void CUPTIAPI launchCallback(void* /* userdata */,
                               CUpti_CallbackDomain domain,
                               CUpti_CallbackId /* cbid */,
                               const void* callbackData) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API) {
      return;
    }
    auto const* data = static_cast<const CUpti_CallbackData*>(callbackData);
    if (data->callbackSite != CUPTI_API_ENTER || data->symbolName == nullptr) {
      return;
    }
    std::string module = t_modules.empty() ? std::string() : t_modules.back();
    std::lock_guard<std::mutex> guard(g_mutex);
    g_records.emplace(data->symbolName, std::move(module));
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
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  enable(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  enable(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel);

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
  t_modules.push_back(mcc.moduleDescription()->moduleLabel());
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
    for (auto const& [kernel, module] : g_records) {
      out << kernel << '\t' << module << '\n';
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
      ->setComment("Path to a text file where the service will write <mangled kernel>\\t<module> lines");
  desc.addUntracked<std::string>("libraryLog", "libraries.txt")
      ->setComment("Path to a text file where the service will write one loaded library path per line");
  descriptions.add("CuptiKernelLoggerService", desc);
}

DEFINE_FWK_SERVICE(CuptiKernelLoggerService);
