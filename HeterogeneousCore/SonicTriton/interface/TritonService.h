#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "oneapi/tbb/concurrent_hash_map.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <functional>
#include <utility>
#include <atomic>
#include <optional>
#include <mutex>

#include "grpc_client.h"

//forward declarations
namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class ProcessContext;
  class ModuleDescription;
  namespace service {
    class SystemBounds;
  }
}  // namespace edm

enum class TritonServerType { Remote = 0, LocalCPU = 1, LocalGPU = 2 };

class TritonService {
public:
  //classes and defs
  struct FallbackOpts {
    FallbackOpts(const edm::ParameterSet& pset)
        : enable(pset.getUntrackedParameter<bool>("enable")),
          debug(pset.getUntrackedParameter<bool>("debug")),
          verbose(pset.getUntrackedParameter<bool>("verbose")),
          container(pset.getUntrackedParameter<std::string>("container")),
          device(pset.getUntrackedParameter<std::string>("device")),
          retries(pset.getUntrackedParameter<int>("retries")),
          wait(pset.getUntrackedParameter<int>("wait")),
          instanceName(pset.getUntrackedParameter<std::string>("instanceName")),
          tempDir(pset.getUntrackedParameter<std::string>("tempDir")),
          imageName(pset.getUntrackedParameter<std::string>("imageName")),
          sandboxDir(pset.getUntrackedParameter<std::string>("sandboxDir")) {
      //randomize instance name
      if (instanceName.empty()) {
        instanceName =
            pset.getUntrackedParameter<std::string>("instanceBaseName") + "_" + edm::createGlobalIdentifier();
      }
    }

    bool enable;
    bool debug;
    bool verbose;
    std::string container;
    std::string device;
    int retries;
    int wait;
    std::string instanceName;
    std::string tempDir;
    std::string imageName;
    std::string sandboxDir;
    std::string command;
  };
  struct Server {
    Server(const edm::ParameterSet& pset)
        : url(pset.getUntrackedParameter<std::string>("address") + ":" +
              std::to_string(pset.getUntrackedParameter<unsigned>("port"))),
          isFallback(pset.getUntrackedParameter<std::string>("name") == fallbackName),
          useSsl(pset.getUntrackedParameter<bool>("useSsl")),
          type(TritonServerType::Remote) {
      if (useSsl) {
        sslOptions.root_certificates = pset.getUntrackedParameter<std::string>("rootCertificates");
        sslOptions.private_key = pset.getUntrackedParameter<std::string>("privateKey");
        sslOptions.certificate_chain = pset.getUntrackedParameter<std::string>("certificateChain");
      }
    }
    Server(const std::string& name_, const std::string& url_, TritonServerType type_)
        : url(url_), isFallback(name_ == fallbackName), useSsl(false), type(type_) {}

    //members
    std::string url;
    bool isFallback;
    bool useSsl;
    TritonServerType type;
    triton::client::SslOptions sslOptions;
    std::unordered_set<std::string> models;
    static const std::string fallbackName;
    static const std::string fallbackAddress;
    static const std::string siteconfName;
  };
  //Dynamic quantities of servers
  struct ServerHealth {
    bool live{false};
    bool ready{false};

    uint64_t inferenceCount{0};
    uint64_t failureCount{0};
    double avgQueueTimeMs{0.0};
    double avgInferTimeMs{0.0};
  };
  struct Model {
    Model(const std::string& path_ = "") : path(path_) {}
    //members
    std::string path;
    std::unordered_set<std::string> servers;
    std::unordered_set<unsigned> modules;
    int refCount{0};  // for dynamic loading on fallback server
    bool isLoaded() const { return refCount > 0; }
  };
  struct Module {
    //currently assumes that a module can only have one associated model
    Module(const std::string& model_) : model(model_) {}
    //members
    std::string model;
  };

  TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);
  ~TritonService() = default;

  //accessors
  void addModel(const std::string& modelName, const std::string& path);

  const std::string* resolveServerName(const std::string& model, const std::string& preferred = "") const;
  const std::pair<const std::string, Server>& resolveServer(const std::string& model,
                                                            const std::string& preferred = "") const;
  std::vector<std::string> unassignedModels() const;

  // update health stats of all servers
  void updateServerHealth(const std::string& modelName = "") const;

  // return the best server for retry, ignore the current server
  std::optional<std::string> getBestServer(const std::string& modelName, const std::string& IgnoreServer = "") const;

  // helper functions to get server statistics?
  //  - getServerSideStatus()
  //  - updateServerStatus()
  //    - loop over servers_ get statistics
  //  - getBestServer(model)
  //    - call updateServerStatus()
  //    - loop over servers_ get their statistics, compute metric, return server name

  const std::string& pid() const { return pid_; }
  void notifyCallStatus(bool status) const;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Dynamic model loading/unloading - only supported for the fallback server
  // The fallback server must be started with explicit model control mode
  // (--model-control-mode explicit) for these functions to work
  bool loadModel(const std::string& modelName);
  bool unloadModel(const std::string& modelName);
  // Start the fallback server if enabled and not already running (idempotent)
  void startFallbackServer();
  bool fallbackStarted() const { return startedFallback_; }

private:
  void preallocate(edm::service::SystemBounds const&);
  void preModuleConstruction(edm::ModuleDescription const&);
  void postModuleConstruction(edm::ModuleDescription const&);
  void preModuleDestruction(edm::ModuleDescription const&);
  void preBeginJob(edm::ProcessContext const&);
  void postEndJob();

  //helper
  template <typename LOG>
  void printFallbackServerLog() const;
  // Internal helpers that operate on Model directly (caller holds lock)
  bool loadModel(const std::string& modelName, Model& model);
  bool unloadModel(const std::string& modelName, Model& model);

  bool verbose_;
  FallbackOpts fallbackOpts_;
  unsigned currentModuleId_;
  bool allowAddModel_;
  bool startedFallback_;
  mutable std::atomic<int> callFails_;
  std::string pid_;
  //this represents a many:many:many map
  std::unordered_map<std::string, Server> servers_;
  //server health needs concurrent-safe edits
  tbb::concurrent_hash_map<std::string, ServerHealth> serversHealth_;
  std::unordered_map<std::string, Model> models_;
  std::unordered_map<unsigned, Module> modules_;
  int numberOfThreads_;

  //Dynamic model loading and unloading (fallback server only)
  std::mutex modelLoadMutex_;
  // Model names currently loaded on the fallback server
  std::unordered_set<std::string> fallbackLoadedModels_;
};

#endif
